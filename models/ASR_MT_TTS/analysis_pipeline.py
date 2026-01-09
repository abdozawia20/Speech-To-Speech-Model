import os
# Disable symlinks warning on Windows for Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
# Add project root to sys.path to allow importing dataset_loader from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import librosa
import soundfile as sf
import evaluate
from datasets import load_dataset
from model import *
from dataset_loader import *
import gc
import json

# Initialize Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
comet = evaluate.load("comet")

def calculate_metrics(predictions, references, sources=None):
    """
    Calculates BLEU, ROUGE, and COMET scores.
    Args:
        predictions: List of hypothesis strings.
        references: List of reference strings.
        sources: List of source strings (required for COMET).
    Returns:
        Dictionary of scores.
    """
    results = {}

    # BLEU
    try:
        bleu_res = bleu.compute(predictions=predictions, references=references)
        results['bleu'] = bleu_res['bleu']
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        results['bleu'] = 0.0

    # ROUGE
    try:
        rouge_res = rouge.compute(predictions=predictions, references=references)
        results['rouge1'] = rouge_res['rouge1']
        results['rouge2'] = rouge_res['rouge2']
        results['rougeL'] = rouge_res['rougeL']
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        results['rouge1'] = 0.0

    # COMET
    # COMET requires 'sources', 'predictions', 'references'
    # If source is same language as reference (e.g. standard ASR), COMET might be weird (it expects MT).
    # But user asked for it. We will pass the 'Target Reference' as 'Source' if real source unavailable, 
    # or better, pass the original Source Text from Language A if we have it?
    # Let's assume 'sources' = Ground Truth Text to treat it as Reference-based eval, 
    # OR we need the actual source text from Lang A.
    if sources:
        try:
            comet_res = comet.compute(predictions=predictions, references=references, sources=sources)
            results['comet'] = comet_res['mean_score']
        except Exception as e:
            print(f"Error computing COMET: {e}")
            results['comet'] = 0.0
    else:
        results['comet'] = None

    return results

def analyze():
    # 1. Load Dataset
    print("Loading Test Data...")

    # data = load_test_data(page=1, page_size=10000) 
    # load_data(start_idx, num_samples, encoding, lang, split, dataset)
    # Loading 'en', 'ar', 'tr' for test split.
    data = load_data(start_idx=0, num_samples=2000, lang=['en', 'de'], split='train')
    
    # 2. Build ID Lookups
    # data is dict: {'en': dataset, 'ar': dataset, ...}
    print("Building ID Lookups...")
    datasets = {}
    id_maps = {}
    
    for lang, ds in data.items():
        datasets[lang] = ds
        # Create map: id -> record
        # Note: 'id' in fleurs is string, ensure consistency
        id_maps[lang] = {}
        try:
             for item in ds:
                 id_maps[lang][str(item['id'])] = item
        except Exception as e:
             print(f"Error processing dataset for {lang}: {e}")

    # 3. Define Pairs
    pairs = [
        ('en', 'de')

        # Add more if needed. 'tr'<->'ar' might not share IDs in FLEURS? 
        # FLEURS is n-way parallel usually.
    ]

    results_log = []

    # 4. Initialize Engines
    # We need a generic way to switch engines.
    # Instantiate them inside loop or keep one global? 
    # STTEngine and TTSEngine seem to handle loading internally.
    
    print("Initializing Engines...")
    # Pre-warm engines if needed
    
    for src_lang, tgt_lang in pairs:
        print(f"\nProcessing Pair: {src_lang} -> {tgt_lang}")
        
        # Filter intersection
        src_ids = set(id_maps[src_lang].keys())
        tgt_ids = set(id_maps[tgt_lang].keys())
        common_ids = sorted(list(src_ids.intersection(tgt_ids)))
        
        if not common_ids:
            print(f"No common matching IDs found for {src_lang}->{tgt_lang}. Skipping.")
            continue
            
        print(f"Found {len(common_ids)} bridged samples.")
        
        # Initialize Engines for this language pair
        # Step 1: STT (Language A) - mostly for completeness/logging
        stt_A = STTEngine(engine="whisper", language=src_lang, model_size="small")
        
        # Step 2: TTS (Language B)
        # Piper specs in STT_TTS_models.py: en, ar, tr supported.
        tts_B = TTSEngine(engine="piper", language=tgt_lang, model_size="low") # or medium

        # Step 3: STT (Language B) - Verifier
        stt_B_Verifier = STTEngine(engine="whisper", language=tgt_lang, model_size="small")

        # Step 4: MT (Language A -> Language B)
        # Using NLLB Small (Int8 quantized if CPU, FP16 if CUDA)
        mt_engine = MTEngine(engine="nllb", model_size="small", src_lang=src_lang, target_lang=tgt_lang)

        tgt_refs = []
        tgt_preds = []
        src_texts = [] # For COMET

        # Limit for demonstration speed
        MAX_SAMPLES = 2000
        print(f"Running pipeline on first {MAX_SAMPLES} samples...")

        for i, cid in enumerate(common_ids[:MAX_SAMPLES]):
            print(f"Processing {i+1}/{len(common_ids[:MAX_SAMPLES])} - ID: {cid}", flush=True)

            # 3. Run Pipeline
            try:
                src_item = id_maps[src_lang][cid]
                tgt_item = id_maps[tgt_lang][cid]

                # 1. Get Source Audio/Text
                src_audio = src_item['audio']['array']
                src_sr = src_item['audio']['sampling_rate']
                src_text_gt = src_item['transcription']

                # 2. Get Target Text (Simulated Translation)
                tgt_text_gt = tgt_item['transcription']
                # A. Run STT on Source
                print(f"  [DEBUG] Starting STT Source...", flush=True)
                text_A_hyp = stt_A.transcribe(src_audio, src_sr)
                print(f"  [STT] Source: {text_A_hyp}")
                
                # B. Translation (NLLB)
                # Use STT hypothesis as input to MT
                print(f"  [DEBUG] Starting MT...", flush=True)
                tgt_text_translated = mt_engine.translate(text_A_hyp)
                print(f"  [MT] {src_lang}->{tgt_lang}: {tgt_text_translated}")

                # C. Run TTS on Translated Text
                # Output of TTS run_inference is ... dict with 'audio' or file path?
                # PiperEngine.run_inference returns {'audio': {'array':..., 'sampling_rate':...}}
                print(f"  [DEBUG] Starting TTS...", flush=True)
                tts_out = tts_B.run_inference(tgt_text_translated)
                
                if not tts_out or 'audio' not in tts_out:
                    print(f"TTS Failed for {cid}")
                    continue
                    
                audio_B_syn = tts_out['audio']['array']
                sr_B_syn = tts_out['audio']['sampling_rate']
                
                # Check for NaNs or excessive values
                has_nan = np.isnan(audio_B_syn).any()
                min_val = np.min(audio_B_syn) if len(audio_B_syn) > 0 else 0
                max_val = np.max(audio_B_syn) if len(audio_B_syn) > 0 else 0
                print(f"  [DEBUG] Audio B Stats: Shape={audio_B_syn.shape}, SR={sr_B_syn}, Min={min_val:.4f}, Max={max_val:.4f}, HasNaN={has_nan}", flush=True)

                if has_nan:
                    print(f"  [WARNING] TTS output contains NaNs. Skipping ID {cid}.")
                    continue

                # D. Run STT on Synthesized Audio
                print(f"  [DEBUG] Starting STT Verifier...", flush=True)
                text_B_rec = stt_B_Verifier.transcribe(audio_B_syn, sr_B_syn)
                print(f"  [DEBUG] Finished STT Verifier.", flush=True)

                # Collect for Metrics
                src_texts.append(src_text_gt)
                tgt_refs.append(tgt_text_gt)
                tgt_preds.append(text_B_rec)

            except Exception as e:
                print(f"Pipeline Error on id {cid}: {e}")
            finally:
                # cleanup
                if 'src_audio' in locals(): del src_audio
                if 'text_A_hyp' in locals(): del text_A_hyp
                if 'tgt_text_translated' in locals(): del tgt_text_translated
                if 'audio_B_syn' in locals(): del audio_B_syn
                if 'text_B_rec' in locals(): del text_B_rec
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Compute Metrics
        if tgt_preds:
            scores = calculate_metrics(tgt_preds, tgt_refs, sources=src_texts)
            scores['pair'] = f"{src_lang}->{tgt_lang}"
            print(f"Scores for {src_lang}->{tgt_lang}: {scores}")
            results_log.append(scores)
        
    # Save Results
    with open("analysis_results.json", "w") as f:
        json.dump(results_log, f, indent=4)
        print("\nResults saved to analysis_results.json")

if __name__ == "__main__":
    analyze()
