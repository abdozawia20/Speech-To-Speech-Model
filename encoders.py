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

    def decode(self, hidden_states: torch.Tensor, speaker_embedding: torch.Tensor, threshold: float = 0.5, minlenratio: float = 0.0, maxlenratio: float = 2.0) -> np.ndarray:
        """
        Reconstruct audio from Wav2Vec2 hidden states using the SpeechT5 decoder.

        The hidden states (768-dim) are projected by SpeechT5's encoder_prenet
        into the model's hidden dim before being passed to the decoder.

        Args:
            hidden_states:     Tensor of shape (1, Seq_Len, 768).
            speaker_embedding: Tensor of shape (512,) or (1, 512).
            threshold, minlenratio, maxlenratio: SpeechT5 generation parameters.

        Returns:
            audio_array: Reconstructed waveform as a numpy float32 array at 16 kHz.

        Raises:
            RuntimeError: If the decoder components were not loaded (load_decoder=False).
        """
        if self.speecht5_model is None or self.vocoder is None:
            raise RuntimeError(
                "Decoder components were not loaded. "
                "Re-initialise with load_decoder=True to use decode()."
            )

        # Ensure (1, Seq_Len, 768)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.to(self.device)

        # Ensure (1, 512)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
        speaker_embedding = speaker_embedding.to(self.device)

        with torch.no_grad():
            # 1. Skip the SpeechT5 feature encoder (conv layers) because they expect
            # raw audio, but we already have Wav2Vec2 hidden states (Seq, 768).
            # We assume dimensions match (Wav2Vec2 base 768 -> SpeechT5 base 768).
            encoder_states = hidden_states.to(self.device)

            # 2. Run SpeechT5 encoder transformer layers on the hidden states
            # In some versions of transformers, the layers are inside 'wrapped_encoder'
            encoder_obj = self.speecht5_model.speecht5.encoder
            if hasattr(encoder_obj, "wrapped_encoder"):
                transformer_encoder = encoder_obj.wrapped_encoder
            else:
                transformer_encoder = encoder_obj

            for layer in transformer_encoder.layers:
                encoder_states = layer(encoder_states)[0]
            
            # 3. Add final layer normalization
            if hasattr(transformer_encoder, "layer_norm") and transformer_encoder.layer_norm is not None:
                encoder_states = transformer_encoder.layer_norm(encoder_states)

            # 4. Manual Autoregressive Decoding Loop
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_states)

            config = self.speecht5_model.config
            num_mel_bins = config.num_mel_bins
            reduction_factor = config.reduction_factor
            
            # Start with zeros
            decoder_input_values = torch.zeros((1, 1, num_mel_bins), device=self.device)
            past_key_values = None
            spectrogram = []
            
            # Heuristic for max length
            max_steps = int(maxlenratio * encoder_states.shape[1] / reduction_factor)
            
            for _ in range(max_steps):
                # Run the decoder prenet on all predicted sequence frames
                # output_sequence: (1, T_mel, 80)
                # decoder_input_values starts with zeros (1, 1, 80)
                # We need to use the model's internal decoder wrapper correctly
                
                decoder_hidden_states = self.speecht5_model.speecht5.decoder.prenet(decoder_input_values, speaker_embedding)
                
                # Wrapped decoder forward
                # We only need the last step if using cache
                decoder_out = self.speecht5_model.speecht5.decoder.wrapped_decoder(
                    hidden_states=decoder_hidden_states[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                
                last_decoder_output = decoder_out.last_hidden_state.squeeze(1) # (1, 768)
                past_key_values = decoder_out.past_key_values
                
                # Predict new mel spectrum
                # feat_out produces (1, reduction_factor * 80)
                spectrum = self.speecht5_model.speech_decoder_postnet.feat_out(last_decoder_output)
                spectrum = spectrum.view(1, reduction_factor, num_mel_bins)
                spectrogram.append(spectrum)
                
                # Next input frame
                # Extract last frame for the next step's input
                new_frame = spectrum[:, -1:, :] # (1, 1, 80)
                decoder_input_values = torch.cat((decoder_input_values, new_frame), dim=1)
                
                # Check stop token
                prob = torch.sigmoid(self.speecht5_model.speech_decoder_postnet.prob_out(last_decoder_output))
                if prob.max() > threshold:
                    break

            # Process collected frames
            if not spectrogram:
                return np.zeros(16000)
                
            # Concatenate along time dimension
            frames = torch.cat(spectrogram, dim=1).flatten(1, 2) # (1, T_total, 80)
            # Apply postbatch (Tacotron style)
            # Actually line 2390 says:
            # spectrograms = model.speech_decoder_postnet.postnet(spectrograms)
            
            # spectrograms in _generate_speech is (Batch, T, 80) after transpose/flatten
            # let's rebuild it similarly
            # spectrogram is list of (1, 2, 80)
            mel_out = torch.stack(spectrogram).transpose(0, 1).flatten(1, 2) # (1, T, 80)
            mel_post = self.speecht5_model.speech_decoder_postnet.postnet(mel_out)
            
            # 5. Pass through vocoder
            speech = self.vocoder(mel_post)
            speech = speech.squeeze()

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
