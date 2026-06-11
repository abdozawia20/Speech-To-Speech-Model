import torch
import numpy as np
import json
import logging
import os
import sys
from tqdm import tqdm
from transformers import pipeline
import evaluate
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cosine

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

OMNI_PHI_DIR = os.path.dirname(os.path.abspath(__file__))
if OMNI_PHI_DIR not in sys.path:
    sys.path.append(OMNI_PHI_DIR)

from model import OmniPhiS2ST
from dataset_loader import load_data

# ── Global configuration ─────────────────────────────────────────────────────
# Change to 2000 for the full benchmark run.
NUM_SAMPLES = 10

ASR_MODEL_ID  = "openai/whisper-base"
SPKREC_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPKREC_SAVEDIR = os.path.join(OMNI_PHI_DIR, "tmp_spkrec")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio array to 16 kHz using librosa (required for ECAPA-TDNN)."""
    if orig_sr == 16000:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=16000)


def analyze(
    num_samples: int = NUM_SAMPLES,
    checkpoint_path: str = None,
    output_file: str = None,
):
    """
    End-to-end evaluation of OmniPhiS2ST on the FLEURS en→de benchmark.

    Pipeline:
        1. Load OmniPhiS2ST from checkpoint_path.
        2. Load FLEURS en_us + de_de via dataset_loader.load_data().
        3. For each matched sample:
               a. model.generate_speech(en_audio, source_sr, max_new_tokens=800) → de_audio (24 kHz)
                  800 tokens @ 150 tok/s ≈ 5.3 s — enough headroom for any FLEURS sentence.
               b. Resample de_audio → 16 kHz for speaker encoder
               c. Whisper ASR(de_audio, language="de") → pred_text
               d. Collect (pred_text, de_ref_text, en_src_text)
               e. ECAPA cosine_similarity(en_audio_emb, de_audio_emb)
        4. Compute BLEU, ROUGE, COMET (with sources), mean cosine similarity.
        5. Save results to output_file (benchmarks/omni_phi.json).

    Args:
        num_samples:      Number of FLEURS samples to load per language.
                          Defaults to NUM_SAMPLES (10 for dev, 2000 for full run).
        checkpoint_path:  Path to fine-tuned OmniPhiS2ST checkpoint directory.
                          Defaults to models/omni_phi/checkpoints/.
        output_file:      Path to write the JSON results file.
                          Defaults to benchmarks/omni_phi.json.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(OMNI_PHI_DIR, "checkpoints")

    if output_file is None:
        output_file = os.path.join(PROJECT_ROOT, "benchmarks", "omni_phi.json")

    logger.info(f"Checkpoint path : {checkpoint_path}")
    logger.info(f"Output file     : {output_file}")
    logger.info(f"Num samples     : {num_samples}")

    if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
        logger.error(
            f"Checkpoint directory '{checkpoint_path}' is empty or does not exist. "
            "Please copy the fine-tuned weights there before running evaluation."
        )
        sys.exit(1)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading OmniPhiS2ST from {checkpoint_path}...")

    model = OmniPhiS2ST(phi4_model_id=checkpoint_path, device=device)
    model.eval()

    # ── 2. Load FLEURS datasets ───────────────────────────────────────────────
    logger.info("Loading FLEURS dataset (en + de)...")
    datasets = load_data(
        lang=["en", "de"],
        split="train",
        num_samples=num_samples,
        dataset=["fleurs"],
    )

    en_ds = datasets.get("en")
    de_ds = datasets.get("de")

    if not en_ds or not de_ds:
        logger.error("Failed to load FLEURS datasets for 'en' or 'de'.")
        sys.exit(1)

    logger.info(f"Loaded {len(en_ds)} en samples, {len(de_ds)} de samples.")
    num_pairs = min(len(en_ds), len(de_ds))

    # ── 3. Initialise ASR (Whisper) ───────────────────────────────────────────
    logger.info("Initialising ASR (Whisper) for transcription of generated audio...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=device,
    )

    # ── 4. Initialise text metrics ────────────────────────────────────────────
    logger.info("Loading metrics (BLEU, ROUGE, COMET)...")
    bleu_metric  = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    comet_metric = None
    try:
        comet_metric = evaluate.load("comet")
        logger.info("COMET metric loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load COMET metric: {e}. COMET will be skipped.")

    # ── 5. Initialise speaker encoder (ECAPA-TDNN) ────────────────────────────
    logger.info("Initialising ECAPA-TDNN speaker encoder for cosine similarity...")
    spk_classifier = None
    try:
        spk_classifier = EncoderClassifier.from_hparams(
            source=SPKREC_SOURCE,
            savedir=SPKREC_SAVEDIR,
            run_opts={"device": device},
        )
    except Exception as e:
        logger.warning(f"Failed to load speaker classifier: {e}. Cosine similarity will be skipped.")

    # ── 6. Inference loop (en → de) ───────────────────────────────────────────
    logger.info("Starting en → de evaluation loop...")

    predictions   = []   # Whisper transcription of generated de audio
    references    = []   # Ground-truth de transcriptions (from FLEURS)
    sources       = []   # Ground-truth en transcriptions (for COMET)
    cos_sims      = []   # ECAPA cosine similarities

    GENERATED_SR  = 24000  # EnCodec decodes at 24 kHz
    partial_file  = output_file + ".partial.json"  # crash-safe incremental save

    def _serialize(obj):
        """JSON serialiser for numpy scalar types."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    for i in tqdm(range(num_pairs), desc="en→de"):
        en_item = en_ds[i]
        de_item = de_ds[i]

        en_src_text = en_item.get("transcription", "")
        de_ref_text = de_item.get("transcription", "")

        en_audio = np.array(en_item["audio"]["array"], dtype=np.float32)
        en_sr    = en_item["audio"]["sampling_rate"]

        try:
            # ── a. Speech-to-speech translation ──────────────────────────────
            # 800 tokens @ 150 tok/s (75 frames/s × 2 codebooks at 1.5 kbps) ≈ 5.3 s.
            # A single generate_speech call is correct here; translate_speech_batched
            # re-translates the same audio on every chunk and would produce
            # nonsense concatenations for sentence-length FLEURS inputs.
            de_audio = model.generate_speech(en_audio, source_sr=en_sr, max_new_tokens=800)
            # de_audio is float32 numpy at 24 kHz

            # ── b. Resample generated audio to 16 kHz for ASR + speaker emb ─
            de_audio_16k = _resample_to_16k(de_audio, orig_sr=GENERATED_SR)

            # ── c. ASR: transcribe generated German audio ─────────────────────
            asr_result = asr_pipeline(
                {"array": de_audio_16k, "sampling_rate": 16000},
                generate_kwargs={"language": "de"},
            )
            pred_text = asr_result["text"].strip()

            # ── d. Accumulate for text metrics ────────────────────────────────
            predictions.append(pred_text)
            references.append(de_ref_text)
            sources.append(en_src_text)

            # ── e. ECAPA cosine similarity ─────────────────────────────────────
            if spk_classifier is not None:
                try:
                    # ECAPA expects [batch, time] tensors at 16 kHz
                    en_audio_16k = _resample_to_16k(en_audio, orig_sr=en_sr)

                    en_tensor = torch.tensor(en_audio_16k).unsqueeze(0).to(device)
                    de_tensor = torch.tensor(de_audio_16k).unsqueeze(0).to(device)

                    with torch.no_grad():
                        en_emb = spk_classifier.encode_batch(en_tensor).squeeze().cpu().numpy()
                        de_emb = spk_classifier.encode_batch(de_tensor).squeeze().cpu().numpy()

                    similarity = 1.0 - cosine(en_emb, de_emb)
                    cos_sims.append(float(similarity))
                except Exception as e:
                    logger.warning(f"Cosine similarity failed at index {i}: {e}")

        except Exception as e:
            logger.error(f"Error at index {i}: {e}")
            continue

        # ── Incremental save every 50 samples (crash-safe for long Colab runs) ─
        if (i + 1) % 50 == 0 or (i + 1) == num_pairs:
            partial = {
                "progress": {"completed": i + 1, "total": num_pairs},
                "predictions": predictions,
                "references":  references,
                "sources":     sources,
                "cos_sims":    cos_sims,
            }
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(partial_file, "w") as _pf:
                json.dump(partial, _pf, indent=2, default=_serialize)
            logger.info(f"[{i+1}/{num_pairs}] Incremental save → {partial_file}")

    # ── 7. Compute metrics ────────────────────────────────────────────────────
    logger.info("Computing metrics...")

    if not predictions:
        logger.error("No predictions collected. Check model output above.")
        sys.exit(1)

    bleu_score  = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    # COMET requires (prediction, reference, source) triples.
    # sources = English ground-truth transcriptions (the original utterance text).
    comet_score = {"mean_score": 0.0}
    if comet_metric is not None:
        try:
            comet_result = comet_metric.compute(
                predictions=predictions,
                references=references,
                sources=sources,
            )
            comet_score = {"mean_score": float(comet_result["mean_score"])}
            logger.info(f"COMET mean score: {comet_score['mean_score']:.4f}")
        except Exception as e:
            logger.error(f"COMET computation failed: {e}")

    avg_cos_sim = float(np.mean(cos_sims)) if cos_sims else 0.0

    # ── 8. Assemble and save results ──────────────────────────────────────────
    final_scores = {
        "en_to_de": {
            "bleu":              bleu_score,
            "rouge":             rouge_score,
            "comet":             comet_score,
            "cosine_similarity": avg_cos_sim,
        }
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(final_scores, f, indent=4, default=_serialize)

    logger.info("Evaluation complete.")
    print(json.dumps(final_scores, indent=4, default=_serialize))

    return final_scores
