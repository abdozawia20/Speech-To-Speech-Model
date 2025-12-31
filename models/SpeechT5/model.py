import torch
import numpy as np
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor, SpeechT5HifiGan
from datasets import load_from_disk
import dataset_loader
import librosa
import sys
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SpeechT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained components
        print("Loading SpeechT5 components...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.model.to(self.device)
        self.vocoder.to(self.device)

        # Set to eval mode by default
        self.model.eval()
        self.vocoder.eval()
        print("Model loaded successfully.")

    # ... (predict method stays same) ...

    def fine_tune(self, source_lang, target_lang, batch_size, epochs, learning_rate):
        """
        Fine-tunes the SpeechT5 model using LoRA.
        """
        print(f"Starting LoRA fine-tuning: {source_lang} -> {target_lang}")
        
        # ... (Data Loading Logic stays same until model setup) ...
        # (Assuming the loading logic is correct and working)
        
        # ... [Data Loading Code Block handled in other chunks if needed, but context shows we are replacing file content to inject LoRA] ...
        # Actually, replace_file_content replaces a single block. I will use multi_replace for safety or target the specific block properly.
        # But wait, I need to add imports at the top.
        # And I need to change the `fine_tune` method implementation heavily around line 179.
        
        pass 

SPEECHT5_N_FFT = 1024
SPEECHT5_HOP_LENGTH = 160  # 10ms at 16k
SPEECHT5_WIN_LENGTH = 400  # 25ms at 16k
SPEECHT5_N_MELS = 80
SPEECHT5_SAMPLING_RATE = 16000

def get_log_mel_spectrogram(audio_array):
    """
    Computes Log-Mel Spectrogram matching SpeechT5 requirements.
    Args:
        audio_array (np.array): Raw audio waveform (16kHz).
    Returns:
        np.array: (Time, 80) log-mel spectrogram.
    """
    # Ensure numpy
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.cpu().numpy()
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, 
        sr=SPEECHT5_SAMPLING_RATE, 
        n_fft=SPEECHT5_N_FFT, 
        hop_length=SPEECHT5_HOP_LENGTH, 
        win_length=SPEECHT5_WIN_LENGTH, 
        n_mels=SPEECHT5_N_MELS,
        fmin=80,
        fmax=7600,
        power=1.0 # Energy (unclear if SpeechT5 uses power 1 or 2, Wav2Vec is usually power 2? No, Tacotron 1.0. Default librosa is 2.0)
        # SpeechT5 uses Kaldi-style filterbanks usually. 
        # But we will use power=2.0 (standard mel) then log.
    )
    
    # Log magnitude (log10(x + 1e-6))
    log_mel_spec = np.log10(mel_spec + 1e-6)
    
    # Transpose to (Time, Channels)
    return log_mel_spec.T.astype(np.float32)
import os


class SpeechT5Dataset(Dataset):
    def __init__(self, source_ds, target_ds, is_preprocessed=True):
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.is_preprocessed = is_preprocessed

    def __len__(self):
        return len(self.source_ds)

    def __getitem__(self, idx):
        src_item = self.source_ds[int(idx)]
        tgt_item = self.target_ds[int(idx)]

        # Handle Source
        src_val = src_item['audio']
        if isinstance(src_val, dict) and 'array' in src_val:
            src_val = src_val['array']
        src_val = np.array(src_val)

        # Handle Target
        tgt_val = tgt_item['audio']
        if isinstance(tgt_val, dict) and 'array' in tgt_val:
            tgt_val = tgt_val['array']
        tgt_val = np.array(tgt_val)

        # Apply preprocessing if not already done
        if len(tgt_val.shape) == 1:
             tgt_val = get_log_mel_spectrogram(tgt_val)

        MAX_AUDIO_LEN = int(5.0 * 16000)
        MAX_SPEC_LEN = int(5.0 * (16000 / SPEECHT5_HOP_LENGTH)) 
        
        if len(src_val.shape) == 1 and len(src_val) > MAX_AUDIO_LEN:
                src_val = src_val[:MAX_AUDIO_LEN]
        
        if len(tgt_val.shape) > 1 and len(tgt_val) > MAX_SPEC_LEN:
            tgt_val = tgt_val[:MAX_SPEC_LEN, :]

        return {
            "input_values": torch.tensor(src_val, dtype=torch.float32),
            "labels": torch.tensor(tgt_val, dtype=torch.float32)
        }

def speecht5_collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return input_values_padded, labels_padded

class SpeechT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained components
        print("Loading SpeechT5 components...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.model.to(self.device)
        self.vocoder.to(self.device)

        # Set to eval mode as we are just running the baseline
        self.model.eval()
        self.vocoder.eval()
        print("Model loaded successfully.")

    def predict(self, audio_array, sampling_rate):
        """
        Runs the SpeechT5 baseline for speech-to-speech conversion.
        
        Args:
            audio_array (np.array): Input audio waveform.
            sampling_rate (int): Sampling rate of the input audio.
            
        Returns:
            np.array: The resulting audio waveform.
        """
        # Prepare the input
        # The processor handles resampling and feature extraction if needed, 
        # but SpeechT5Processor for VC usually expects audio input.
        inputs = self.processor(audio=audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Check if input has 'input_values' or 'input_features' depending on processor specifics
        # The SpeechT5Processor usually returns input_values for audio
        
        # Generate dummy speaker embedding (1, 512)
        # SpeechT5 requires speaker embeddings for generating speech
        speaker_embeddings = torch.randn((1, 512)).to(self.device)

        # Generate speech
        # We pass the input_values from the processor
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_values"], 
                speaker_embeddings, 
                vocoder=self.vocoder
            )

        return speech.cpu().numpy()

    def fine_tune(self, source_lang, target_lang, batch_size, epochs, learning_rate, resume_from_checkpoint=None):
        """
        Fine-tunes the SpeechT5 model.

        Args:
            source_lang (str): Source language code (e.g., 'en').
            target_lang (str): Target language code (e.g., 'tr').
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
        """
        print(f"Starting fine-tuning: {source_lang} -> {target_lang}")
        
        # 1. Load Data
        print("Loading datasets...")
        
        # Construct path for preprocessed data
        # Mapping de-en etc.
        # We assume the preprocessing script saved as 'datasets/processed_speecht5_{source}_{target}'
        # Note: The pair naming in dataset_loader was 'deA-enA'. 
        # But our manual script used 'datasets/processed_speecht5_de_en'.
        # Let's check for that specific pattern first.
        
        preprocessed_path = os.path.join(dataset_loader.DATASETS_DIR, f"processed_speecht5_{source_lang}_{target_lang}")
        preprocessed_path_rev = os.path.join(dataset_loader.DATASETS_DIR, f"processed_speecht5_{target_lang}_{source_lang}")
        
        source_ds = None
        target_ds = None
        is_preprocessed = False
        
        # Check both permutations
        if os.path.exists(preprocessed_path):
            active_path = preprocessed_path
        elif os.path.exists(preprocessed_path_rev):
            active_path = preprocessed_path_rev
        else:
            active_path = None
        
        if active_path:
            print(f"Found preprocessed data at {active_path}. Loading...")
            try:
                source_ds = load_from_disk(os.path.join(active_path, source_lang))
                target_ds = load_from_disk(os.path.join(active_path, target_lang))
                is_preprocessed = True
                print("Successfully loaded preprocessed spectrograms.")
            except Exception as e:
                print(f"Failed to load preprocessed data: {e}. Falling back to raw load.")
        
        if not is_preprocessed:
             print("Loading raw data via dataset_loader (On-the-fly processing will be used)...")
             datasets = dataset_loader.load_data(
                lang=[source_lang, target_lang], 
                split="train",
                dataset=['seamless_align'] # Explicitly request Seamless Align
             )
             source_ds = datasets.get(source_lang)
             target_ds = datasets.get(target_lang)

        if not source_ds or not target_ds:
            raise ValueError(f"Could not load datasets for {source_lang} and {target_lang}")
        
        # Ensure lengths match
        min_len = min(len(source_ds), len(target_ds))
        if len(source_ds) != len(target_ds):
            print(f"Warning: Dataset lengths differ ({len(source_ds)} vs {len(target_ds)}). Truncating to {min_len}.")
            # We must truncate to ensure index alignment matches
            source_ds = source_ds.select(range(min_len))
            target_ds = target_ds.select(range(min_len))
        
        num_samples = min_len

        # --- LoRA Configuration ---
        # SpeechT5 is an Encoder-Decoder model. We target the attention layers.
        # Common targets: "q_proj", "v_proj" (for both encoder and decoder usually)
        peft_config = LoraConfig(
            # task_type=TaskType.SEQ_2_SEQ_LM, # Removed: causes AttributeError on SpeechT5
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"] # Target attention-related projections
        )
        
        # Wrap the model for LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Handle resume from checkpoint
        start_epoch = 0
        if resume_from_checkpoint:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            try:
                # Load the adapter weights
                self.model = PeftModel.from_pretrained(self.model, resume_from_checkpoint)
                # Optionally, load optimizer state and epoch number if saved
                # For simplicity, we'll just resume training from epoch 0 with loaded weights.
                # A more robust checkpointing system would save optimizer state and current epoch.
                print("LoRA adapters loaded successfully from checkpoint.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        
        self.model.train()
        
        # Note: Gradient Checkpointing can be enabled with LoRA if needed for extra savings,
        # but with R=16 LoRA, it might fit without it.
        # If we enable it, ensure we do: self.model.enable_input_require_grads()
        # self.model.gradient_checkpointing_enable() 
        # For now, let's try WITHOUT checkpointing to avoid the "Backward Second Time" headache.
        # LoRA itself drastically reduces memory for optimizer states.
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Simple training loop
        # num_samples was calculated above
        
        try:
            # Create Dataset and DataLoader
            # Ensure we pass the possibly truncated datasets
            train_dataset = SpeechT5Dataset(source_ds, target_ds, is_preprocessed=is_preprocessed)
            
            # pinned_memory=True allows faster copy to CUDA
            # num_workers > 0 enables background data loading
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=speecht5_collate_fn,
                num_workers=4, 
                pin_memory=True
            )

            for epoch in range(start_epoch, epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                epoch_loss = 0.0
                num_batches = 0
                
                # Progress bar for the batch loop
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
                
                for input_values_padded, labels_padded in pbar:
                    # Move to device
                    input_values_padded = input_values_padded.to(self.device)
                    labels_padded = labels_padded.to(self.device)
                    
                    batch_curr_size = input_values_padded.size(0)
                    speaker_embeddings = torch.randn((batch_curr_size, 512)).to(self.device)

                    # Forward
                    optimizer.zero_grad()
                    
                    outputs = self.model(
                        input_values=input_values_padded,
                        speaker_embeddings=speaker_embeddings,
                        labels=labels_padded
                    )
                    
                    loss = outputs.loss
                    if loss is None:
                        pred = outputs.spectrogram
                        target = labels_padded
                        if pred.shape != target.shape:
                            min_len = min(pred.shape[1], target.shape[1])
                            pred = pred[:, :min_len, :]
                            target = target[:, :min_len, :]
                        loss = torch.nn.functional.l1_loss(pred, target)

                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                    # Update progress bar
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                 
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        except KeyboardInterrupt:
            print("\nInterrupt received! Saving training state...")
            save_path = f"speecht5_{source_lang}_{target_lang}_interrupted"
            
            # Save Adapters
            self.model.save_pretrained(save_path)
            
            # Save Training State
            state = {
                "epoch": epoch,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size
            }
            with open(os.path.join(save_path, "training_state.json"), "w") as f:
                json.dump(state, f)
            print(f"State saved to {save_path}. Resume by passing this path to resume_checkpoint.")
            sys.exit(0)
            
        print("Fine-tuning complete.")

    def save(self, path="./FineTunedSpeechT5"):
        """Saves the fine-tuned model, processor, and vocoder."""
        print(f"Saving components to {path}...")
        os.makedirs(path, exist_ok=True)
        
        # Save each component in a subdirectory to avoid file collisions (e.g. config.json)
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        self.vocoder.save_pretrained(os.path.join(path, "vocoder"))
        print("Saved.")

    def load(self, path):
        """Loads the model, processor, and vocoder from path."""
        print(f"Loading components from {path}...")
        
        # Load from subdirectories
        model_path = os.path.join(path, "model")
        processor_path = os.path.join(path, "processor")
        vocoder_path = os.path.join(path, "vocoder")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")

        # Load Processor and Vocoder
        self.processor = SpeechT5Processor.from_pretrained(processor_path)
        try:
             self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_path)
        except Exception as e:
             print(f"Warning: Could not load vocoder from {vocoder_path}: {e}")
             print("Keeping default/current vocoder.")

        # Load Model (LoRA vs Full)
        # Check if it's a LoRA adapter
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Detected LoRA adapters. Loading base model + adapters...")
            # Reload base model just in case (to ensure clean state)
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
            self.model.to(self.device)
            
            # Load adapters
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print("Detected full model. Loading...")
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path)
            self.model.to(self.device)
        
        self.model.eval()
        self.vocoder.eval()
        print("Loaded.")
