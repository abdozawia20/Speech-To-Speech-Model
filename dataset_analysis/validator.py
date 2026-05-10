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

    def run(self, num_samples=None):
        print(f"Loading FLEURS for languages: {TARGET_LANGUAGES}")
        datasets = dataset_loader.load_data(
            lang=TARGET_LANGUAGES,
            split="train",
            dataset=["fleurs"],
            num_samples=num_samples
        )
        
        # Ensure we have both languages
        if not all(lang in datasets for lang in TARGET_LANGUAGES):
            print("Error: Could not load both languages.")
            return

        en_ds = datasets['en']
        de_ds = datasets['de']
        
        count = 0
        for i in range(len(en_ds)):
            sample_id = str(en_ds[i]['id'])
            
            if sample_id in self.results:
                continue

            en_audio = en_ds[i]['audio']['array']
            de_audio = de_ds[i]['audio']['array']
            
            en_ref = en_ds[i]['transcription']
            de_ref = de_ds[i]['transcription']

            # Transcribe
            en_asr = self.transcribe(en_audio)
            de_asr = self.transcribe(de_audio)

            # Calculate WER
            en_wer = self.calculate_wer(en_ref, en_asr)
            de_wer = self.calculate_wer(de_ref, de_asr)

            passed = (en_wer <= WER_THRESHOLD) and (de_wer <= WER_THRESHOLD)

            self.results[sample_id] = {
                "en_wer": round(en_wer, 4),
                "de_wer": round(de_wer, 4),
                "passed": passed
            }

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} new samples. Last ID: {sample_id} | EN WER: {en_wer:.2f} | DE WER: {de_wer:.2f}")
                self.save_results()

        self.save_results()
        self.print_summary()

    def print_summary(self):
        total = len(self.results)
        if total == 0:
            print("\nNo results to summarize.")
            return
            
        passed = sum(1 for r in self.results.values() if r['passed'])
        avg_en = sum(r['en_wer'] for r in self.results.values()) / total
        avg_de = sum(r['de_wer'] for r in self.results.values()) / total
        
        print("\n--- Validation Summary ---")
        print(f"Total processed: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {total - passed}")
        print(f"Avg EN WER: {avg_en:.4f}")
        print(f"Avg DE WER: {avg_de:.4f}")

if __name__ == "__main__":
    validator = FleursSemanticValidator()
    # Run validation
    validator.run(num_samples=None)
