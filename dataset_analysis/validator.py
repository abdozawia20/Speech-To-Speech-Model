import os
import json
import torch
import numpy as np
from faster_whisper import WhisperModel
import jiwer
import sys

# Ensure root is in path for dataset_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataset_loader

# Configuration
TARGET_LANGUAGES = ['en', 'de']
WER_THRESHOLD = 0.10
MODEL_SIZE = 'base'
RESULTS_FILE = 'dataset_analysis/fleurs_validation_results.json'

class FleursSemanticValidator:
    def __init__(self, model_size=MODEL_SIZE):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(model_size, device=device, compute_type="float32" if device == "cpu" else "float16")
        self.results = {}
        self.load_results()

    def load_results(self):
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} existing records.")

    def save_results(self):
        tmp_file = RESULTS_FILE + ".tmp"
        with open(tmp_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        os.rename(tmp_file, RESULTS_FILE)

    def normalize_text(self, text):
        return jiwer.transforms.Compose([
            jiwer.transforms.ToLowerCase(),
            jiwer.transforms.RemovePunctuation(),
            jiwer.transforms.RemoveMultipleSpaces(),
            jiwer.transforms.Strip(),
        ])(text)

    def calculate_wer(self, reference, hypothesis):
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)
        if not ref: return 1.0 # Avoid division by zero
        return jiwer.wer(ref, hyp)

    def transcribe(self, audio_array):
        segments, _ = self.model.transcribe(audio_array, beam_size=5)
        return " ".join([segment.text for segment in segments])
