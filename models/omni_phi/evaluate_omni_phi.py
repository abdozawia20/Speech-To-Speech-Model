import torch
import numpy as np
import json
import logging
import os
import sys
import argparse
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
from lang_config import get_whisper_lang, LANG_CONFIG

# ── Global configuration ─────────────────────────────────────────────────────
# Change to 2000 for the full benchmark run.
NUM_SAMPLES = 10

ASR_MODEL_ID   = "openai/whisper-base"
SPKREC_SOURCE  = "speechbrain/spkrec-ecapa-voxceleb"
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
    lang_tgt: str = "de",
):
    """
    End-to-end evaluation of OmniPhiS2ST on the FLEURS en→{lang_tgt} benchmark.

    Pipeline:
        1. Load OmniPhiS2ST from checkpoint_path (defaults to checkpoints_en_{lang_tgt}/).
        2. Load FLEURS en_us + {lang_tgt} via dataset_loader.load_data().
        3. For each matched sample:
               a. model.generate_speech(en_audio, source_sr, max_new_tokens=800) → tgt_audio (24 kHz)
                  800 tokens @ 150 tok/s ≈ 5.3 s — enough headroom for any FLEURS sentence.
               b. Resample tgt_audio → 16 kHz for speaker encoder
               c. Whisper ASR(tgt_audio, language=whisper_lang) → pred_text
               d. Collect (pred_text, tgt_ref_text, en_src_text)
               e. ECAPA cosine_similarity(en_audio_emb, tgt_audio_emb)
        4. Compute BLEU, ROUGE, COMET (with sources), mean cosine similarity.
        5. Save results to output_file (benchmarks/omni_phi_en_{lang_tgt}.json).

    Args:
        num_samples:      Number of FLEURS samples to load per language.
                          Defaults to NUM_SAMPLES (10 for dev, 2000 for full run).
        checkpoint_path:  Path to fine-tuned OmniPhiS2ST checkpoint directory.
                          Defaults to models/omni_phi/checkpoints_en_{lang_tgt}/.
        output_file:      Path to write the JSON results file.
                          Defaults to benchmarks/omni_phi_en_{lang_tgt}.json.
        lang_tgt:         Target language ISO prefix (e.g. 'de', 'fr', 'it').
                          Must match a key in lang_config.LANG_CONFIG.
    """
    # ── Validate lang_tgt ─────────────────────────────────────────────────────
    if lang_tgt not in LANG_CONFIG:
        supported = ", ".join(sorted(LANG_CONFIG.keys()))
        logger.error(
            f"Unsupported lang_tgt '{lang_tgt}'. Supported languages: {supported}."
        )
        sys.exit(1)

    whisper_lang = get_whisper_lang(lang_tgt)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(OMNI_PHI_DIR, f"checkpoints_en_{lang_tgt}")

    if output_file is None:
        output_file = os.path.join(PROJECT_ROOT, "benchmarks", f"omni_phi_en_{lang_tgt}.json")

    logger.info(f"Language pair   : en → {lang_tgt}  (Whisper tag: '{whisper_lang}')")
    logger.info(f"Checkpoint path : {checkpoint_path}")
    logger.info(f"Output file     : {output_file}")
    logger.info(f"Num samples     : {num_samples}")

    if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
        logger.error(
            f"Checkpoint directory '{checkpoint_path}' is empty or does not exist. "
            f"Please copy the fine-tuned weights there before running evaluation.\n"
            f"Expected location: {checkpoint_path}"
        )
        sys.exit(1)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading OmniPhiS2ST from {checkpoint_path}...")

    model = OmniPhiS2ST(
        phi4_model_id=checkpoint_path,
        device=device,
        lang_prefix=lang_tgt,
    )
    model.eval()

    # ── 2. Load FLEURS datasets ───────────────────────────────────────────────
    logger.info(f"Loading FLEURS dataset (en + {lang_tgt})...")
    datasets = load_data(
        lang=["en", lang_tgt],
        split="train",
        num_samples=num_samples,
        dataset=["fleurs"],
    )

    en_ds  = datasets.get("en")
    tgt_ds = datasets.get(lang_tgt)

    if not en_ds or not tgt_ds:
        logger.error(f"Failed to load FLEURS datasets for 'en' or '{lang_tgt}'.")
        sys.exit(1)

    logger.info(f"Loaded {len(en_ds)} en samples, {len(tgt_ds)} {lang_tgt} samples.")
    num_pairs = min(len(en_ds), len(tgt_ds))

    # ── 3. Initialise ASR (Whisper) ───────────────────────────────────────────
    logger.info(f"Initialising ASR (Whisper) for transcription (language='{whisper_lang}')...")
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

    # ── 6. Inference loop (en → lang_tgt) ─────────────────────────────────────
    logger.info(f"Starting en → {lang_tgt} evaluation loop...")

    predictions   = []   # Whisper transcription of generated target audio
    references    = []   # Ground-truth target transcriptions (from FLEURS)
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

    for i in tqdm(range(num_pairs), desc=f"en→{lang_tgt}"):
        en_item  = en_ds[i]
        tgt_item = tgt_ds[i]

        en_src_text  = en_item.get("transcription", "")
        tgt_ref_text = tgt_item.get("transcription", "")

        en_audio = np.array(en_item["audio"]["array"], dtype=np.float32)
        en_sr    = en_item["audio"]["sampling_rate"]

        try:
            # ── a. Speech-to-speech translation ──────────────────────────────
            # 800 tokens @ 150 tok/s (75 frames/s × 2 codebooks at 1.5 kbps) ≈ 5.3 s.
            # A single generate_speech call is correct here; translate_speech_batched
            # re-translates the same audio on every chunk and would produce
            # nonsense concatenations for sentence-length FLEURS inputs.
            tgt_audio = model.generate_speech(en_audio, source_sr=en_sr, max_new_tokens=800)
            # tgt_audio is float32 numpy at 24 kHz

            # ── b. Resample generated audio to 16 kHz for ASR + speaker emb ─
            tgt_audio_16k = _resample_to_16k(tgt_audio, orig_sr=GENERATED_SR)

            # ── c. ASR: transcribe generated target audio ──────────────────────
            asr_result = asr_pipeline(
                {"array": tgt_audio_16k, "sampling_rate": 16000},
                generate_kwargs={"language": whisper_lang},
            )
            pred_text = asr_result["text"].strip()

            # ── d. Accumulate for text metrics ────────────────────────────────
            predictions.append(pred_text)
            references.append(tgt_ref_text)
            sources.append(en_src_text)

            # ── e. ECAPA cosine similarity ─────────────────────────────────────
            if spk_classifier is not None:
                try:
                    # ECAPA expects [batch, time] tensors at 16 kHz
                    en_audio_16k = _resample_to_16k(en_audio, orig_sr=en_sr)

                    en_tensor  = torch.tensor(en_audio_16k).unsqueeze(0).to(device)
                    tgt_tensor = torch.tensor(tgt_audio_16k).unsqueeze(0).to(device)

                    with torch.no_grad():
                        en_emb  = spk_classifier.encode_batch(en_tensor).squeeze().cpu().numpy()
                        tgt_emb = spk_classifier.encode_batch(tgt_tensor).squeeze().cpu().numpy()

                    similarity = 1.0 - cosine(en_emb, tgt_emb)
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
    pair_key = f"en_to_{lang_tgt}"
    final_scores = {
        pair_key: {
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OmniPhiS2ST on the FLEURS benchmark for a given target language."
    )
    parser.add_argument(
        "--lang_tgt", default="de",
        help="Target language ISO prefix, e.g. 'de', 'fr', 'it' (default: 'de').",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to fine-tuned checkpoint directory. "
             "Defaults to models/omni_phi/checkpoints_en_{lang_tgt}/.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write the JSON results file. "
             "Defaults to benchmarks/omni_phi_en_{lang_tgt}.json.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=NUM_SAMPLES,
        help=f"Number of FLEURS samples to evaluate (default: {NUM_SAMPLES}).",
    )
    args = parser.parse_args()

    analyze(
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint,
        output_file=args.output,
        lang_tgt=args.lang_tgt,
    )
