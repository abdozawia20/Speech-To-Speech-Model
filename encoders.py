import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, EncodecModel, AutoProcessor, SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan


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


class Wav2VecSpeechT5Encoder(nn.Module):
    """
    A Wav2Vec2-based encoder designed to be compatible with the SpeechT5 model.

    - encode(): Runs raw audio through facebook/wav2vec2-base-960h and returns
      hidden states of shape (Batch, Seq_Len, 768). These hidden states can be
      fed directly into SpeechT5's encoder projection layer.

    - decode(): Takes hidden states and a speaker embedding and uses the SpeechT5
      decoder + HiFi-GAN vocoder to reconstruct audio. Useful for unit testing
      and verifying the quality of encoded representations.

    Note: This encoder is frozen by default (feature extraction only).
    Note: The decode() functionality requires SpeechT5ForSpeechToSpeech components.
    """

    WAV2VEC_MODEL = "facebook/wav2vec2-base-960h"
    SPEECHT5_MODEL = "microsoft/speecht5_vc"
    VOCODER_MODEL  = "microsoft/speecht5_hifigan"

    def __init__(self, wav2vec_model_name: str = WAV2VEC_MODEL, speecht5_model_name: str = SPEECHT5_MODEL, vocoder_model_name: str  = VOCODER_MODEL, load_decoder: bool = True,
    ):
        super(Wav2VecSpeechT5Encoder, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ------------------------------------------------------------------ #
        # Wav2Vec2 Encoder (frozen)                                           #
        # ------------------------------------------------------------------ #
        print(f"[Wav2VecSpeechT5Encoder] Loading Wav2Vec2 processor from '{wav2vec_model_name}'...")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)

        print(f"[Wav2VecSpeechT5Encoder] Loading Wav2Vec2 model from '{wav2vec_model_name}'...")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).to(self.device)

        # Freeze encoder weights — we use it purely for feature extraction.
        for param in self.wav2vec_model.parameters():
            param.requires_grad = False
        self.wav2vec_model.eval()

        # ------------------------------------------------------------------ #
        # SpeechT5 Decoder + Vocoder (optional — only for decode())          #
        # ------------------------------------------------------------------ #
        self.speecht5_model = None
        self.vocoder = None

        if load_decoder:
            print(f"[Wav2VecSpeechT5Encoder] Loading SpeechT5 model from '{speecht5_model_name}'...")
            self.speecht5_processor = SpeechT5Processor.from_pretrained(speecht5_model_name)
            self.speecht5_model = SpeechT5ForSpeechToSpeech.from_pretrained(speecht5_model_name).to(self.device)
            self.speecht5_model.eval()

            print(f"[Wav2VecSpeechT5Encoder] Loading HiFi-GAN vocoder from '{vocoder_model_name}'...")
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_name).to(self.device)
            self.vocoder.eval()

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def encode(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Encode a single raw-audio waveform into Wav2Vec2 hidden states.

        Args:
            audio_array: 1-D numpy array of float32 waveform samples.
            sr:          Sampling rate of the waveform (will be resampled to 16 kHz
                         via the processor if needed).

        Returns:
            hidden_states: Tensor of shape (1, Seq_Len, 768) on CPU.
        """
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Wav2Vec2Processor normalises the signal; resample if needed
        inputs = self.wav2vec_processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self.device)   # (1, T)

        with torch.no_grad():
            outputs = self.wav2vec_model(input_values)

        # Shape: (1, Seq_Len, 768)  — returned to CPU so it can be stored/serialised
        return outputs.last_hidden_state.cpu()

    def encode_batch(self, audio_arrays: list, sr: int = 16000) -> list:
        """
        Encode a list of variable-length raw-audio waveforms.

        Args:
            audio_arrays: List of 1-D numpy arrays.
            sr:           Common sampling rate for all arrays.

        Returns:
            List of numpy arrays, each of shape (Seq_Len, 768).
        """
        flat = [np.array(a, dtype=np.float32).flatten() for a in audio_arrays]

        inputs = self.wav2vec_processor(
            flat,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,           # pad to longest in batch
        )
        input_values  = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None

        with torch.no_grad():
            outputs = self.wav2vec_model(
                input_values,
                attention_mask=attention_mask,
            )

        # (Batch, Seq_Len, 768) → list of (Seq_Len_i, 768) numpy arrays (unpadded)
        hidden_states = outputs.last_hidden_state.cpu()  # (B, S, 768)
        result = []
        for i in range(hidden_states.size(0)):
            result.append(hidden_states[i].numpy())       # (S, 768)
        return result

    def decode(self, hidden_states: torch.Tensor, speaker_embedding: torch.Tensor,
               threshold: float = 0.5, minlenratio: float = 0.0,
               maxlenratio: float = 2.0) -> np.ndarray:
        """
        Reconstruct audio from Wav2Vec2 hidden states using the SpeechT5 decoder.

        Uses transformer_enc.forward() — matching model.py:_encode_wav2vec_states —
        so LayerNorm, dropout, and relative position_bias are applied consistently
        with training.

        Args:
            hidden_states:     Tensor  (1, Seq_Len, 768).
            speaker_embedding: Tensor  (512,)  or  (1, 512).
            threshold:         Stop-token probability threshold (lower → longer output).
            minlenratio, maxlenratio: Output length relative to encoder seq length.

        Returns:
            Reconstructed waveform as a numpy float32 array at 16 kHz.
        """
        if self.speecht5_model is None or self.vocoder is None:
            raise RuntimeError(
                "Decoder components were not loaded. "
                "Re-initialise with load_decoder=True to use decode()."
            )

        # ── normalise input shapes ────────────────────────────────────────── #
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)      # (1, S, 768)
        hidden_states = hidden_states.to(self.device)

        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)   # (1, 512)
        speaker_embedding = speaker_embedding.to(self.device)

        with torch.no_grad():
            # ── 1. SpeechT5 transformer encoder ──────────────────────────── #
            # Use full forward() so LayerNorm, dropout & position_bias are
            # applied — same as model.py:_encode_wav2vec_states.
            encoder_obj = self.speecht5_model.speecht5.encoder
            transformer_enc = (
                encoder_obj.wrapped_encoder
                if hasattr(encoder_obj, "wrapped_encoder")
                else encoder_obj
            )

            # All tokens real (no padding in single-sample inference)
            enc_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.long, device=self.device
            )
            enc_out = transformer_enc(
                hidden_states=hidden_states,
                attention_mask=enc_mask,
                return_dict=True,
            )
            encoder_states = enc_out.last_hidden_state   # (1, S, 768)

            # ── 2. Autoregressive decoding loop ──────────────────────────── #
            config = self.speecht5_model.config
            num_mel_bins     = config.num_mel_bins      # 80
            reduction_factor = config.reduction_factor  # 2

            enc_len   = encoder_states.shape[1]
            min_steps = max(0, int(minlenratio * enc_len / reduction_factor))
            max_steps = (
                int(maxlenratio * enc_len / reduction_factor)
                if maxlenratio > 0 else 1000
            )
            max_steps = max(max_steps, min_steps + 1, 1)

            # Start with a single all-zero frame
            dec_input = torch.zeros((1, 1, num_mel_bins), device=self.device)
            past_kv  = None
            frames   = []   # list of (1, reduction_factor, 80)

            for step in range(max_steps):
                dec_hidden = self.speecht5_model.speecht5.decoder.prenet(
                    dec_input, speaker_embedding
                )
                dec_out = self.speecht5_model.speecht5.decoder.wrapped_decoder(
                    hidden_states=dec_hidden[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=None,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                last_h  = dec_out.last_hidden_state.squeeze(1)   # (1, 768)
                past_kv = dec_out.past_key_values

                # (1, RF * 80) → (1, RF, 80)
                spec = self.speecht5_model.speech_decoder_postnet.feat_out(last_h)
                spec = spec.view(1, reduction_factor, num_mel_bins)
                frames.append(spec)

                dec_input = torch.cat([dec_input, spec[:, -1:, :]], dim=1)

                # stop-token (honoured only after min_steps)
                # prob_out returns (1, reduction_factor) — use .max() not .item()
                if step >= min_steps:
                    stop_p = torch.sigmoid(
                        self.speecht5_model.speech_decoder_postnet.prob_out(last_h)
                    )
                    if stop_p.max() > threshold:
                        break

            # ── 3. Assemble mel, apply postnet, vocoder ───────────────────── #
            if not frames:
                return np.zeros(16000, dtype=np.float32)

            # frames: list of (1, RF, 80)  →  (1, T, 80)
            mel_out  = torch.cat(frames, dim=1)   # (1, N*RF, 80)
            mel_post = self.speecht5_model.speech_decoder_postnet.postnet(mel_out)
            speech   = self.vocoder(mel_post).squeeze()

        return speech.cpu().numpy()


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


# --------------------------------------------------------------------------- #
# Renamed from SpeechT5Encoder for clarity: this class produces              #
# log-mel spectrograms (80-bin) using the SpeechT5 feature extractor.        #
# --------------------------------------------------------------------------- #
class SpeechT5MelSpectrogramEncoder(nn.Module):
    """
    Wraps the SpeechT5Processor to produce 80-bin log-mel spectrograms,
    the native target-side representation for SpeechT5 models.

    Formerly named: SpeechT5Encoder
    """
    def __init__(self, model_name="microsoft/speecht5_tts"):
        super(SpeechT5MelSpectrogramEncoder, self).__init__()
        # SpeechT5Processor handles the log-mel spectrogram extraction
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        
    def encode(self, audio_array, sr=16000):
        inputs = self.processor(audio=audio_array, sampling_rate=sr)
        return inputs.input_values


# Backwards-compatible alias so existing code that imports SpeechT5Encoder still works
SpeechT5Encoder = SpeechT5MelSpectrogramEncoder
