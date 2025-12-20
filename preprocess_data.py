import dataset_loader
from encoders import SpectogramEncoder
import os

def main():
    # Configuration
    TOTAL_SAMPLES = 10000
    CHUNK_SIZE = 100  # Process 100 samples at a time
    LANGUAGES = ['tr']
    OUTPUT_DIR = "./processed_data"

    print(f"Initializing SpectogramEncoder...")
    encoder = SpectogramEncoder()

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    for start_idx in range(0, TOTAL_SAMPLES, CHUNK_SIZE):
        current_chunk_size = min(CHUNK_SIZE, TOTAL_SAMPLES - start_idx)
        print(f"\n--- Processing Chunk: start_idx={start_idx}, size={current_chunk_size} ---")

        try:
            datasets = dataset_loader.load_train_data(
                start_idx=start_idx,
                num_samples=current_chunk_size,
                encoding=encoder,
                lang=LANGUAGES
            )

            # Check if any data was processed in this chunk
            samples_processed = 0
            for lang, ds in datasets.items():
                if ds is not None and len(ds) > 0:
                    samples_processed += len(ds)
                    # Create a subdirectory for the language if it doesn't exist
                    lang_dir = os.path.join(OUTPUT_DIR, lang)
                    if not os.path.exists(lang_dir):
                        os.makedirs(lang_dir)
                    
                    # Save the chunk
                    # Naming convention: {lang}_chunk_{start_idx}_{end_idx}
                    end_idx = start_idx + len(ds)
                    chunk_name = f"chunk_{start_idx}_{end_idx}"
                    save_path = os.path.join(lang_dir, chunk_name)
                    
                    print(f"Saving {lang} chunk to {save_path}...")
                    ds.save_to_disk(save_path)
                    print(f"Successfully saved {len(ds)} samples for {lang} to {chunk_name}.")
                else:
                    print(f"No data found for language: {lang} in this chunk.")
            
            if samples_processed == 0:
                print(f"No samples found for any language in this chunk. Stopping processing as dataset seems exhausted.")
                break
                    
        except Exception as e:
            print(f"Error processing chunk starting at {start_idx}: {e}")
            # Continue to next chunk
            continue

    print("\nData processing complete.")

if __name__ == "__main__":
    main()
