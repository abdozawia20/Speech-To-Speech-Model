import sys
import os
import traceback

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(project_root)

from dataset_loader import load_data

def test_cvss_loading():
    print("Testing CVSS data loading...")
    try:
        # Load a small number of samples from CVSS for French-English
        # We expect 'fr' and 'en' datasets to be returned and aligned.
        datasets = load_data(num_samples=5, dataset=['cvss'], lang=['de', 'en'])
        
        if not datasets:
            print("No datasets returned.")
            return

        for lang, ds in datasets.items():
            if ds:
                print(f"Dataset '{lang}' length: {len(ds)}")
            else:
                print(f"Dataset '{lang}' is None.")

        if 'de' not in datasets or 'en' not in datasets:
            print("Missing 'de' or 'en' in returned datasets.")
            return
            
        ds_de = datasets['de']
        ds_en = datasets['en']
        
        if ds_de is None or ds_en is None:
            print("One of the datasets is None.")
            return

        # 1. Verify same length
        if len(ds_de) != len(ds_en):
            print(f"FAILED: Length mismatch! de: {len(ds_de)}, en: {len(ds_en)}")
        else:
            print("SUCCESS: Lengths match.")

        # 2. Check IDs are identical and sorted
        ids_de = ds_de['id']
        ids_en = ds_en['id']
        
        if ids_de == ids_en:
            print("SUCCESS: IDs match exactly.")
        else:
            print("FAILED: IDs do not match!")
            print(f"DE IDs: {ids_de}")
            print(f"EN IDs: {ids_en}")

        is_sorted = all(ids_de[i] <= ids_de[i+1] for i in range(len(ids_de)-1))
        if is_sorted:
            print("SUCCESS: IDs are sorted.")
        else:
            print("FAILED: IDs are NOT sorted.")

        # 3. Print transcription of the first sample
        if len(ds_de) > 0:
            print("\nSample 1 Pair:")
            print(f"DE Transcription: {ds_de[0]['transcription']}")
            print(f"EN Transcription: {ds_en[0]['transcription']}")
            print(f"DE ID: {ds_de[0]['id']}")
            print(f"EN ID: {ds_en[0]['id']}")
        else:
            print("No samples to display.")

    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "403" in error_msg or "access" in error_msg.lower():
            print("\n--- HF GATED ACCESS DETECTED ---")
            print("It seems you don't have access to mozilla-foundation/common_voice_4_0.")
            print("Please ensure you are logged in using `huggingface-cli login` and have accepted the terms on the dataset page.")
            print(f"Error details: {error_msg}")
        else:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_cvss_loading()
