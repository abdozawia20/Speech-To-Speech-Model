"""
Speech-to-Speech Inference API
===============================
Exposes the following endpoints:

    POST /infer/speecht5        — Audio A -> Audio B via fine-tuned SpeechT5
    POST /infer/speecht5_wavlm  — Audio A -> Audio B via SpeechT5 + WavLM encoder
    POST /infer/asr_mt_tts      — Audio A -> Audio B via cascaded ASR -> MT -> TTS pipeline

Memory management:
    Only ONE model is active in memory at any time.
    All incoming requests are queued (FIFO) via an asyncio.Lock so that a new
    model is never loaded until the previous inference has finished and the
    previous model has been evicted from GPU/CPU memory.
"""

import asyncio
import gc
import io
import logging
import os
import sys
import tempfile
import uuid

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse

# -------------------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sts-api")

# -------------------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports work correctly
# regardless of where uvicorn is launched from.
# -------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------------------------
# App
# -------------------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "Inference",
        "description": "Endpoints for running speech-to-speech inference models.",
    },
    {
        "name": "System",
        "description": "System health and status check.",
    },
]

app = FastAPI(
    title="Speech-to-Speech Inference API",
    description=(
        "Run inference across all implemented speech-to-speech models.\n\n"
        "### Key Features:\n"
        "* **Model Isolation**: Only one model is active in memory at a time to prevent OOM.\n"
        "* **FIFO Queuing**: Requests are handled sequentially using a global lock.\n"
        "* **Multi-Model Support**: Direct support for SpeechT5, SpeechT5+WavLM, and cascaded pipelines."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects to the Swagger UI documentation."""
    return RedirectResponse(url="/docs")

# -------------------------------------------------------------------------------------
# Model Manager
# -------------------------------------------------------------------------------------

class ModelManager:
    """
    Ensures only one model is resident in memory at a time.

    Usage (inside an endpoint):
        async with model_manager.lock:
            model = model_manager.load("speecht5")
            result = model.run_inference(...)
            # Lock is released after the `async with` block exits.
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self._current_name: str | None = None
        self._current_model = None

    def load(self, name: str):
        """
        Return the requested model, evicting any previously loaded model first.
        This method is synchronous and must be called *inside* the async lock.
        """
        if self._current_name == name:
            return self._current_model

        # Evict the previous model from memory
        self._evict()

        logger.info(f"Loading model: {name}")
        model = self._instantiate(name)
        self._current_name = name
        self._current_model = model
        return model

    def _evict(self):
        if self._current_model is not None:
            logger.info(f"Evicting model: {self._current_name}")
            del self._current_model
            self._current_model = None
            self._current_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _instantiate(name: str):
        """Lazily import and construct the requested model."""
        if name == "speecht5":
            from models.SpeechT5.model import SpeechT5
            # Use the wav2vec v2 checkpoint (30 epochs, Mar 24) — the most
            # complete fine-tuned checkpoint available. The spectrogram checkpoint
            # (speecht5_en_de_spectrogram) is a partial/interrupted save from Jan 1
            # and does not produce German output reliably.
            model = SpeechT5(encoder_type="wav2vec")
            ckpt_path = os.path.join(PROJECT_ROOT, "models", "SpeechT5", "speecht5_en_de_wav2vec_v2")
            logger.info(f"Loading weights from: {ckpt_path}")
            model.load(ckpt_path)
            if model.target_embeddings is None:
                logger.info("Fetching German speaker embedding from Fleurs...")
                model.get_speaker_embedding('de')
            else:
                logger.info("Using speaker embedding loaded from checkpoint.")
            return model

        if name == "speecht5_wavlm":
            from models.SpeechT5WavLM.model import SpeechT5WavLM
            model = SpeechT5WavLM()
            ckpt_path = os.path.join(PROJECT_ROOT, "models", "SpeechT5WavLM", "speecht5_wavlm_en_de_v2")
            logger.info(f"Loading weights from: {ckpt_path}")
            model.load(ckpt_path)
            if model.target_embeddings is None:
                logger.info("Fetching German speaker embedding from Fleurs...")
                model.get_speaker_embedding('de')
            else:
                logger.info("Using speaker embedding loaded from checkpoint.")
            return model

        if name == "asr_mt_tts":
            # Return a lightweight container for the three sub-engines.
            # The engines are instantiated here so they are also evicted as
            # one unit when the model manager switches to a different model.
            from models.ASR_MT_TTS.model import STTEngine, MTEngine, TTSEngine

            class _ASRMTPipeline:
                def __init__(self):
                    # Default language pair: English -> German (hardcoded engines,
                    # language pair is passed at inference time via `translate`).
                    # We defer per-language engine construction to run_inference
                    # so the manager only stores one pipeline object.
                    self._stt_cache: dict = {}
                    self._mt_cache: dict = {}
                    self._tts_cache: dict = {}

                def _get_stt(self, lang: str):
                    if lang not in self._stt_cache:
                        self._stt_cache[lang] = STTEngine(
                            engine="whisper", language=lang, model_size="small"
                        )
                    return self._stt_cache[lang]

                def _get_mt(self, src_lang: str, tgt_lang: str):
                    key = f"{src_lang}_{tgt_lang}"
                    if key not in self._mt_cache:
                        self._mt_cache[key] = MTEngine(
                            engine="nllb",
                            model_size="small",
                            src_lang=src_lang,
                            target_lang=tgt_lang,
                        )
                    return self._mt_cache[key]

                def _get_tts(self, lang: str):
                    if lang not in self._tts_cache:
                        self._tts_cache[lang] = TTSEngine(
                            engine="piper", language=lang, model_size="low"
                        )
                    return self._tts_cache[lang]

                def run_inference(
                    self,
                    audio_array: np.ndarray,
                    sampling_rate: int,
                    src_lang: str,
                    tgt_lang: str,
                ) -> dict:
                    # Step 1: ASR
                    stt = self._get_stt(src_lang)
                    transcript = stt.transcribe(audio_array, sampling_rate)
                    logger.info(f"[ASR_MT_TTS] Transcript: {transcript}")

                    # Step 2: MT
                    mt = self._get_mt(src_lang, tgt_lang)
                    translation = mt.translate(transcript)
                    logger.info(f"[ASR_MT_TTS] Translation: {translation}")

                    # Step 3: TTS
                    tts = self._get_tts(tgt_lang)
                    result = tts.run_inference(translation)

                    if result is None or "audio" not in result:
                        raise RuntimeError("TTS engine returned no audio.")

                    return result  # {'audio': {'array': ..., 'sampling_rate': ...}}

            return _ASRMTPipeline()

        raise ValueError(f"Unknown model name: '{name}'")


model_manager = ModelManager()


# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------

def _read_audio(upload: UploadFile) -> tuple[np.ndarray, int]:
    """Read an uploaded audio file and return (array, sample_rate)."""
    raw_bytes = upload.file.read()
    with io.BytesIO(raw_bytes) as buf:
        audio_array, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
    
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # to mono
        
    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        
    return audio_array, sample_rate


def _remove_file(path: str):
    """Attempt to delete a file from the filesystem."""
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Deleted temporary file: {path}")
    except Exception as e:
        logger.error(f"Failed to delete temporary file {path}: {e}")


def _array_to_wav_response(
    array: np.ndarray, sample_rate: int, background_tasks: BackgroundTasks
) -> FileResponse:
    """
    Write *array* to a temporary WAV file and return it as a FileResponse.
    The file is written to /tmp with a unique name so concurrent requests
    do not collide. Cleanup is handled by BackgroundTasks.
    """
    tmp_path = os.path.join(tempfile.gettempdir(), f"sts_out_{uuid.uuid4().hex}.wav")
    sf.write(tmp_path, array, sample_rate)
    
    # Add a background task to delete the file after the response is sent
    background_tasks.add_task(_remove_file, tmp_path)
    
    return FileResponse(
        path=tmp_path,
        media_type="audio/wav",
        filename="output.wav",
    )


# -------------------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------------------

@app.post(
    "/infer/speecht5",
    summary="SpeechT5 (Spectrogram encoder) inference",
    response_description="Synthesized speech as a WAV file",
    tags=["Inference"],
)
async def infer_speecht5(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Source audio file (any format supported by libsndfile)"),
):
    """
    Translate speech using the fine-tuned **SpeechT5** model with a default
    spectrogram encoder. Upload an audio file and receive back a synthesized WAV file in
    the target language.
    """
    audio_array, sample_rate = _read_audio(audio)

    async with model_manager.lock:
        model = model_manager.load("speecht5")
        result = model.run_inference(audio_array, sample_rate)

    if result is None or "audio" not in result:
        raise HTTPException(status_code=500, detail="Model returned no audio output.")

    return _array_to_wav_response(
        result["audio"]["array"], result["audio"]["sampling_rate"], background_tasks
    )


@app.post(
    "/infer/speecht5_wavlm",
    summary="SpeechT5 + WavLM encoder inference",
    response_description="Synthesized speech as a WAV file",
    tags=["Inference"],
)
async def infer_speecht5_wavlm(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Source audio file (any format supported by libsndfile)"),
):
    """
    Translate speech using the hybrid **SpeechT5 + WavLM** model. WavLM
    hidden states are used as encoder features, bypassing SpeechT5's native
    CNN front-end. Upload an audio file and receive back a synthesized WAV
    file in the target language.
    """
    audio_array, sample_rate = _read_audio(audio)

    async with model_manager.lock:
        model = model_manager.load("speecht5_wavlm")
        result = model.run_inference(audio_array, sample_rate)

    if result is None or "audio" not in result:
        raise HTTPException(status_code=500, detail="Model returned no audio output.")

    return _array_to_wav_response(
        result["audio"]["array"], result["audio"]["sampling_rate"], background_tasks
    )


@app.post(
    "/infer/asr_mt_tts",
    summary="Cascaded ASR → MT → TTS pipeline",
    response_description="Synthesized speech as a WAV file",
    tags=["Inference"],
)
async def infer_asr_mt_tts(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Source audio file (any format supported by libsndfile)"),
    src_lang: str = Form(..., description="BCP-47 source language code, e.g. 'en'"),
    tgt_lang: str = Form(..., description="BCP-47 target language code, e.g. 'de'"),
):
    """
    Translate speech using the **cascaded ASR → MT → TTS** pipeline:

    1. **Whisper** (small) transcribes the source audio.
    2. **NLLB** (small, 600M) translates the transcript.
    3. **Piper** synthesizes the translated text into speech.

    Upload a source audio file, specify the source and target language codes,
    and receive back a synthesized WAV file.
    """
    audio_array, sample_rate = _read_audio(audio)

    async with model_manager.lock:
        pipeline_model = model_manager.load("asr_mt_tts")
        try:
            result = pipeline_model.run_inference(
                audio_array, sample_rate, src_lang=src_lang, tgt_lang=tgt_lang
            )
        except Exception as exc:
            logger.error(f"Inference failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _array_to_wav_response(
        result["audio"]["array"], result["audio"]["sampling_rate"], background_tasks
    )


# -------------------------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------------------------

@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Returns 200 OK if the server is running."""
    return {"status": "ok"}
