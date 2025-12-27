import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, EncodecModel, AutoProcessor

class SpectrogramEncoder:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def encode(self, audio_array, sr=None):
        if sr is None:
            sr = self.sample_rate

        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        S_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return S_db

    def decode(self, S_db, n_iter=32):
        S_power = librosa.db_to_power(S_db)

        audio_reconstructed = librosa.feature.inverse.mel_to_audio(
            S_power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=n_iter
        )
        
        return audio_reconstructed

class Wav2VecEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super(Wav2VecEncoder, self).__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load the Processor (Handles resampling and normalization logic)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # 2. Load the Model (The neural network with pretrained weights)
        # We use Wav2Vec2Model (not ForCTC) because you want the vectors/hidden states, not text.
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)

        # Freezing the model (Optional):
        # If you only want to EXTRACT features and not train the wav2vec part,
        # un-comment the loop below. For now, we leave it trainable or strictly for extraction.
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, input_values):
        # Move inputs to the same device as the model
        if isinstance(input_values, torch.Tensor):
            input_values = input_values.to(self.device)
            
        # Pass inputs through the model
        outputs = self.model(input_values)

        # .last_hidden_state contains the vector representations (Z)
        # Shape: (Batch_Size, Sequence_Length, Hidden_Size)
        return outputs.last_hidden_state

class VQGANEncoder(nn.Module):
    def __init__(self, model_name="facebook/encodec_24khz"):
        super(VQGANEncoder, self).__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load the Processor (Handles resampling logic internally)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # 2. Load the Model (This acts as the VQ-GAN)
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)

    def encode(self, input_values, bandwidth=None):
        # Move inputs to the same device as the model
        if isinstance(input_values, torch.Tensor):
            input_values = input_values.to(self.device)
            
        # Turns Audio -> Discrete Codes
        # Result shape: (Batch, Frames, Codebooks)
        return self.model.encode(input_values, bandwidth=bandwidth)

    def decode(self, audio_codes, audio_scales=None):
        # Turns Discrete Codes -> Audio
        # We generally don't need scales for simple reconstruction in recent versions,
        # but the model expects the argument structure.
        return self.model.decode(audio_codes, audio_scales)[0]
