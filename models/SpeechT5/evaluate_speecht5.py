import torch
import numpy as np
import json
import logging
import os
import sys
from tqdm import tqdm
from transformers import pipeline
import evaluate
from comet import download_model, load_from_checkpoint

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.SpeechT5.model import SpeechT5
from dataset_loader import load_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze(num_samples=2000, model_path=None, output_file=None):
    # Configuration
    # Model is in the same directory as this script by default
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speecht5_en_de_partially_trained")
    
    if output_file is None:
        output_file = os.path.join(PROJECT_ROOT, "speecht5.json")
        
    MODEL_PATH = model_path
    NUM_SAMPLES_PER_LANG = num_samples
    ASR_MODEL_ID = "openai/whisper-base"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load SpeechT5 Model
    logger.info(f"Loading SpeechT5 model from {MODEL_PATH}...")
    speech_model = SpeechT5()
    
    # Check if model path exists
    full_model_path = MODEL_PATH
    if not os.path.exists(full_model_path):
        logger.error(f"Model path {full_model_path} does not exist!")
        sys.exit(1)
        
    speech_model.load(full_model_path)
    speech_model.model.to(device)
    speech_model.vocoder.to(device)

    # 2. Load Datasets (Fleurs En & De)
    logger.info("Loading Fleurs dataset (En & De)...")
    
    datasets = load_data(
        lang=['en', 'de'], 
        split='train', 
        num_samples=NUM_SAMPLES_PER_LANG, 
        dataset=['fleurs']
    )
    
    if not datasets.get('en') or not datasets.get('de'):
        logger.error("Failed to load datasets for 'en' or 'de'.")
        sys.exit(1)

    # 3. Initialize ASR for Transcription (Measuring text quality from generated speech)
    logger.info("Initializing ASR (Whisper) for usage in metrics...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=ASR_MODEL_ID, 
        device=device
    )

    # 4. Initialize Metrics
    logger.info("Loading metrics (BLEU, ROUGE, COMET)...")
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    try:
        comet_metric = evaluate.load("comet") 
    except Exception as e:
        logger.warning(f"Failed to load COMET via evaluate: {e}. Trying raw comet check...")
        comet_metric = None

    # Store results
    results = {
        "en_to_de": {"references": [], "predictions": [], "sources": []},
        "de_to_en": {"references": [], "predictions": [], "sources": []}
    }

    # Helper for inference loop
    def evaluate_pair(src_lang, tgt_lang, src_ds, tgt_ds):
        logger.info(f"Evaluating {src_lang} -> {tgt_lang}...")
        
        limit = min(len(src_ds), len(tgt_ds))
        
        # Get target speaker embedding once
        speech_model.get_speaker_embedding(tgt_lang)
        
        for i in tqdm(range(limit), desc=f"{src_lang}->{tgt_lang}"):
            src_item = src_ds[i]
            tgt_item = tgt_ds[i]
            
            src_text = src_item['transcription']
            ref_text = tgt_item['transcription']
            
            src_audio = src_item['audio']['array']
            src_sr = src_item['audio']['sampling_rate']
            
            try:
                out = speech_model.run_inference(
                    src_audio, 
                    src_sr, 
                    speaker_embedding=speech_model.target_embeddings
                )
                pred_audio = out['audio']['array']
                pred_sr = out['audio']['sampling_rate']
                
                transcription = asr_pipeline(
                    {"array": pred_audio, "sampling_rate": pred_sr}, 
                    generate_kwargs={"language": tgt_lang}
                )["text"]
                
                key = f"{src_lang}_to_{tgt_lang}"
                results[key]["predictions"].append(transcription)
                results[key]["references"].append(ref_text)
                results[key]["sources"].append(src_text)
                
            except Exception as e:
                logger.error(f"Error at index {i}: {e}")
                continue

    # Run Bi-directional
    evaluate_pair('en', 'de', datasets['en'], datasets['de'])
    evaluate_pair('de', 'en', datasets['de'], datasets['en'])

    # 5. Compute Metrics
    logger.info("Computing metrics...")
    final_scores = {}

    for direction, data in results.items():
        preds = data["predictions"]
        refs = data["references"]
        srcs = data["sources"]
        
        if not preds:
            continue
            
        bleu_score = bleu.compute(predictions=preds, references=refs)
        rouge_score = rouge.compute(predictions=preds, references=refs)
        
        comet_score = {"mean_score": 0.0}
        if comet_metric:
            try:
                comet_res = comet_metric.compute(predictions=preds, references=refs, sources=srcs)
                comet_score = {"mean_score": comet_res["mean_score"]}
            except Exception as e:
                logger.error(f"COMET computation failed for {direction}: {e}")
        
        final_scores[direction] = {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "comet": comet_score
        }

    # 6. Save Results
    logger.info(f"Saving results to {output_file}...")
    
    def serialize(obj):
        if isinstance(obj, np.float32): return float(obj)
        if isinstance(obj, np.float64): return float(obj)
        raise TypeError
        
    with open(output_file, 'w') as f:
        json.dump(final_scores, f, indent=4, default=serialize)
    
    logger.info("Evaluation Complete.")
    print(json.dumps(final_scores, indent=4, default=serialize))