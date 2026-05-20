import os
import json
import sys

# Ensure root is in path for dataset_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataset_loader

RESULTS_FILE = 'dataset_analysis/fleurs_validation_results.json'
OUTPUT_NAME = 'speech_t5_perfected'

def create_perfected_dataset():
    """
    Extracts samples that passed semantic validation and saves them
    as a new dataset ready for training/preprocessing.
    """
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Run the validator first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Get IDs that passed (keys in JSON are strings)
    passed_ids = {str(uid) for uid, data in results.items() if data.get('passed')}
    print(f"Found {len(passed_ids)} samples that passed validation.")

    if not passed_ids:
        print("No samples passed validation. Aborting.")
        return

    # Load FLEURS datasets via dataset_loader
    print("Loading original FLEURS dataset...")
    # Note: load_data will perform its own validation/uniqueness check
    # and return aligned datasets.
    datasets = dataset_loader.load_data(
        lang=['en', 'de'],
        split="train",
        dataset=["fleurs"],
        num_samples=None
    )

    # Filter each language for the passed IDs
    for lang in datasets:
        ds = datasets[lang]
        # Filter indices where the 'id' (converted to string) is in passed_ids
        # We use str(uid) because passed_ids came from JSON keys (strings)
        indices = [i for i, uid in enumerate(ds['id']) if str(uid) in passed_ids]
        datasets[lang] = ds.select(indices)
        print(f"Perfected {lang}: {len(datasets[lang])} samples.")

    # Save to disk
    out_path = os.path.join(dataset_loader.DATASETS_DIR, OUTPUT_NAME)
    os.makedirs(out_path, exist_ok=True)
    
    for lang in datasets:
        lang_path = os.path.join(out_path, lang)
        print(f"Saving {lang} perfected dataset to {lang_path}...")
        datasets[lang].save_to_disk(lang_path)

    print(f"\nSUCCESS! Perfected dataset created at: {out_path}")
    print(f"Structure:")
    print(f"  {OUTPUT_NAME}/en/")
    print(f"  {OUTPUT_NAME}/de/")

if __name__ == "__main__":
    create_perfected_dataset()
