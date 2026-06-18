import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model, EncodecModel, AutoProcessor, SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, WavLMModel

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


# Monkey-patch torchaudio for SpeechBrain compatibility
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# Monkey-patch torch.amp.custom_fwd for SpeechBrain 1.1.0 compatibility
if not hasattr(torch, "amp"):
    # Create a dummy amp module if it doesn't exist (very old torch)
    class DummyAMP:
        def custom_fwd(self, fwd=None, **kwargs):
            return fwd if fwd is not None else lambda f: f
    torch.amp = DummyAMP()
elif not hasattr(torch.amp, "custom_fwd"):
    if hasattr(torch.cuda.amp, "custom_fwd"):
        def _custom_fwd_patch(fwd=None, **kwargs):
            kwargs.pop("device_type", None)
            return torch.cuda.amp.custom_fwd(fwd, **kwargs)
        torch.amp.custom_fwd = _custom_fwd_patch
    else:
        torch.amp.custom_fwd = lambda fwd=None, **kwargs: fwd if fwd is not None else lambda f: f

class Wav2VecSpeechT5Encoder(nn.Module):
    """
    Supports English, German, Italian, and French by leveraging the SpeechT5 Speech-to-Speech unified latent space.
    Replaces the English-specific Wav2Vec2 with the SpeechT5 Encoder.
    """
    ST5_MODEL = "microsoft/speecht5_vc"
    VOCODER_MODEL = "microsoft/speecht5_hifigan"

    def __init__(self, wav2vec_model_name: str = ST5_MODEL, speecht5_model_name: str = ST5_MODEL, vocoder_model_name: str = VOCODER_MODEL, load_decoder: bool = True):
        super(Wav2VecSpeechT5Encoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Unified Processor
        self.processor = SpeechT5Processor.from_pretrained(speecht5_model_name)
        
        # 2. Load Unified Model (Contains both Encoder and Decoder)
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(speecht5_model_name).to(self.device)
        
        self.vocoder = None
        if load_decoder:
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_name).to(self.device)
            self.vocoder.eval()
            
        self.model.eval()

        # Speaker recognition for embedding extraction
        self.spk_classifier = None

    def get_speaker_embedding(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Extracts a 512-dim x-vector from the provided audio.
        """
        if self.spk_classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self.spk_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="tmp_spkrec",
                run_opts={"device": self.device}
            )
        
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().to(self.device)
            # Add batch dimension
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            embeddings = self.spk_classifier.encode_batch(audio_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.squeeze(1) # (batch, 512)
        
        return embeddings.cpu()

    def encode(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """ Extracts hidden states that the decoder ACTUALLY understands. """
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
            
        inputs = self.processor(audio=audio_array, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # We use the internal speecht5 encoder specifically
            encoder_outputs = self.model.speecht5.encoder(
                input_values=inputs.input_values,
                attention_mask=inputs.attention_mask
            )
        return encoder_outputs.last_hidden_state.cpu() # (1, Seq_Len, 768)

    def encode_batch(self, audio_arrays: list, sr: int = 16000) -> list:
        """
        Encode a list of variable-length raw-audio waveforms.
        """
        flat = [np.array(a, dtype=np.float32).flatten() for a in audio_arrays]

        inputs = self.processor(
            audio=flat,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values  = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None

        with torch.no_grad():
            outputs = self.model.speecht5.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        hidden_states = outputs.last_hidden_state.cpu()  # (B, S, 768)
        result = []
        for i in range(hidden_states.size(0)):
            result.append(hidden_states[i].numpy())       # (S, 768)
        return result

    def decode(self, hidden_states: torch.Tensor, speaker_embedding: torch.Tensor, **kwargs) -> np.ndarray:
        """
        Decodes hidden states into a speech waveform using the SpeechT5 decoder.
        The latent space is shared with the encoder, eliminating the 'drone' artifact.
        Mirrors the official `_generate_speech` function from the transformers library.
        """
        if self.vocoder is None:
             raise RuntimeError("Vocoder not loaded. Initialize with load_decoder=True.")
             
        # ── normalise input shapes ────────────────────────────────────────── #
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)      # (1, S, 768)
        hidden_states = hidden_states.to(self.device)

        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)   # (1, 512)
        speaker_embedding = speaker_embedding.to(self.device)

        with torch.no_grad():
            # Manual autoregressive decoding loop to bypass redundant encoding.
            # This logic mirrors transformers.models.speecht5.modeling_speecht5._generate_speech
            config = self.model.config
            num_mel_bins = config.num_mel_bins
            reduction_factor = config.reduction_factor
            
            # Use thresholds/ratios from kwargs or defaults.
            # NOTE: maxlenratio default is 20.0 (matching _generate_speech), NOT 2.0.
            # A value of 2.0 causes extremely early termination and clipped audio.
            #
            # For long inputs (>10s), speecht5_vc's cross-attention may only cover
            # ~50% of encoder frames before the stop token fires, producing audio
            # that is shorter than the source. Setting disable_stop_token=True forces
            # generation up to maxlenratio and is recommended for reconstruction use.
            threshold = kwargs.get("threshold", 0.5)
            minlenratio = kwargs.get("minlenratio", 0.0)
            maxlenratio = kwargs.get("maxlenratio", 20.0)
            if kwargs.get("disable_stop_token", False):
                threshold = 1.1  # above sigmoid max — stop token never fires

            enc_len = hidden_states.shape[1]
            min_steps = int(minlenratio * enc_len / reduction_factor)
            # When stop token is disabled, use a tighter maxlenratio so the decoder
            # doesn't run all 1875 positional-encoding steps. The natural 1:1 ratio
            # (one decoder step per encoder frame) is maxlenratio = RF = 2.0; we add
            # 20% headroom → 1.2 * enc_len / RF  ≈  enc_len * 0.6 steps.
            effective_maxlenratio = maxlenratio
            if kwargs.get("disable_stop_token", False) and maxlenratio > 2.0:
                effective_maxlenratio = kwargs.get("stop_disabled_maxlenratio", 1.2)
            max_steps = int(effective_maxlenratio * enc_len / reduction_factor)
            # Clamp to the decoder's positional encoding limit (config.max_speech_positions).
            # output_sequence starts at length 1 and grows by 1 each step, so the
            # absolute maximum is (max_speech_positions - 1) steps.
            pos_limit = config.max_speech_positions - 1
            max_steps = max(min(max_steps, pos_limit), min_steps + 1, 1)

            # Start the output sequence with a mel spectrum that is all zeros.
            output_sequence = hidden_states.new_zeros(1, 1, num_mel_bins)
            past_kv  = None
            spectrogram   = []

            for step in range(max_steps):
                # Run the decoder prenet on the entire output sequence.
                dec_hidden = self.model.speecht5.decoder.prenet(output_sequence, speaker_embedding)
                # Run the decoder layers on the last element of the prenet output.
                dec_out = self.model.speecht5.decoder.wrapped_decoder(
                    hidden_states=dec_hidden[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=None,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                last_h  = dec_out.last_hidden_state.squeeze(1)
                past_kv = dec_out.past_key_values

                # Predict the new mel spectrum for this step in the sequence.
                spec = self.model.speech_decoder_postnet.feat_out(last_h)
                spec = spec.view(1, reduction_factor, num_mel_bins)
                spectrogram.append(spec)

                # Extend the output sequence with the new mel spectrum.
                new_frame = spec[:, -1, :].view(1, 1, num_mel_bins)
                output_sequence = torch.cat((output_sequence, new_frame), dim=1)

                # Predict the probability that this is the stop token.
                # Use max over reduction_factor frames: stop when the most confident
                # sub-frame exceeds the threshold (Tacotron-2 semantics).
                # NOTE: The HF _generate_speech uses sum(), but with reduction_factor=2
                # that fires at avg per-frame prob of 0.25 — far too sensitive, causing
                # output to be ~50% shorter than the source. max() is correct here.
                prob = torch.sigmoid(self.model.speech_decoder_postnet.prob_out(last_h))
                if step >= min_steps and torch.max(prob, dim=-1).values >= threshold:
                    break

            if not spectrogram:
                return np.zeros(16000, dtype=np.float32)

            # Assemble mel, apply postnet, vocoder
            mel_out = torch.stack(spectrogram).transpose(0, 1).flatten(1, 2)  # (1, T * RF, 80)
            
            # The postnet method already handles transposition and the residual connection
            mel_post = self.model.speech_decoder_postnet.postnet(mel_out)
            
            speech = self.vocoder(mel_post).squeeze()

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
        # NOTE: Using audio_target=... is crucial to get mel spectrograms
        # instead of normalized raw audio.
        inputs = self.processor(audio_target=audio_array, sampling_rate=sr)
        return inputs.input_values


# Backwards-compatible alias so existing code that imports SpeechT5Encoder still works
SpeechT5Encoder = SpeechT5MelSpectrogramEncoder

class WavLMSpeechT5Encoder(nn.Module):
    """
    A WavLM-based encoder designed to be compatible with the SpeechT5 model.
    Functions similarly to Wav2VecSpeechT5Encoder, but utilizes microsoft/wavlm-base-plus.
    """

    WAVLM_MODEL = "microsoft/wavlm-base-plus"
    SPEECHT5_MODEL = "microsoft/speecht5_vc"
    VOCODER_MODEL  = "microsoft/speecht5_hifigan"

    def __init__(self, wavlm_model_name: str = WAVLM_MODEL, speecht5_model_name: str = SPEECHT5_MODEL, vocoder_model_name: str  = VOCODER_MODEL, load_decoder: bool = True):
        super(WavLMSpeechT5Encoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[WavLMSpeechT5Encoder] Loading Wav2Vec2FeatureExtractor for '{wavlm_model_name}'...")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)

        print(f"[WavLMSpeechT5Encoder] Loading WavLM model from '{wavlm_model_name}'...")
        self.wavlm_model = WavLMModel.from_pretrained(wavlm_model_name).to(self.device)

        for param in self.wavlm_model.parameters():
            param.requires_grad = False
        self.wavlm_model.eval()

        self.speecht5_model = None
        self.vocoder = None

        if load_decoder:
            print(f"[WavLMSpeechT5Encoder] Loading SpeechT5 model from '{speecht5_model_name}'...")
            self.speecht5_processor = SpeechT5Processor.from_pretrained(speecht5_model_name)
            self.speecht5_model = SpeechT5ForSpeechToSpeech.from_pretrained(speecht5_model_name).to(self.device)
            self.speecht5_model.eval()

            print(f"[WavLMSpeechT5Encoder] Loading HiFi-GAN vocoder from '{vocoder_model_name}'...")
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_name).to(self.device)
            self.vocoder.eval()

        # Speaker recognition model — loaded lazily on first call to get_speaker_embedding
        self.spk_classifier = None

    def get_speaker_embedding(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Extracts a 512-dim x-vector from the provided audio using SpeechBrain.
        Identical API to Wav2VecSpeechT5Encoder.get_speaker_embedding.
        """
        if self.spk_classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self.spk_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="tmp_spkrec",
                run_opts={"device": self.device}
            )

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().to(self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            embeddings = self.spk_classifier.encode_batch(audio_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.squeeze(1)  # (batch, 512)

        return embeddings.cpu()

    def encode(self, audio_array: np.ndarray, sr: int = 16000) -> torch.Tensor:
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        inputs = self.wavlm_processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            outputs = self.wavlm_model(input_values)
        return outputs.last_hidden_state.cpu()

    def encode_batch(self, audio_arrays: list, sr: int = 16000) -> list:
        flat = [np.array(a, dtype=np.float32).flatten() for a in audio_arrays]
        inputs = self.wavlm_processor(flat, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values  = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
        with torch.no_grad():
            outputs = self.wavlm_model(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.cpu()
        result = []
        for i in range(hidden_states.size(0)):
            result.append(hidden_states[i].numpy())
        return result

    def decode(self, hidden_states: torch.Tensor, speaker_embedding: torch.Tensor,
               threshold: float = 0.5, minlenratio: float = 0.0, maxlenratio: float = 20.0,
               disable_stop_token: bool = False) -> np.ndarray:
        """
        Decodes WavLM hidden states into a speech waveform using the SpeechT5 decoder.

        WavLM features are injected directly into the cross-attention of the SpeechT5
        decoder, bypassing the SpeechT5 encoder entirely. This avoids a latent space
        mismatch that would occur from re-processing WavLM features through the
        SpeechT5 encoder's self-attention layers.

        Args:
            disable_stop_token: If True, the stop-token check is bypassed and the
                decoder runs until maxlenratio. Recommended for long inputs where the
                cross-attention may not traverse the full encoder sequence.

        NOTE: maxlenratio default is 20.0 (matching the official _generate_speech),
        NOT 2.0. A value of 2.0 causes extremely early termination.
        """
        if self.speecht5_model is None or self.vocoder is None:
            raise RuntimeError("Decoder components were not loaded.")

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.to(self.device)

        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
        speaker_embedding = speaker_embedding.to(self.device)

        with torch.no_grad():
            # Feed WavLM hidden states DIRECTLY into the decoder cross-attention.
            # Do NOT pass them through speecht5_model.speecht5.encoder (wrapped_encoder),
            # as that adds a spurious self-attention projection that corrupts the
            # WavLM latent space and causes severe audio quality degradation.
            encoder_states = hidden_states

            config = self.speecht5_model.config
            num_mel_bins = config.num_mel_bins
            reduction_factor = config.reduction_factor

            enc_len = encoder_states.shape[1]
            min_steps = max(0, int(minlenratio * enc_len / reduction_factor))
            # When stop token is disabled, use a tighter maxlenratio so the decoder
            # doesn't run all 1875 positional-encoding steps.
            effective_maxlenratio = maxlenratio
            if disable_stop_token and maxlenratio > 2.0:
                effective_maxlenratio = 1.2  # ~1:1 with 20% headroom
            max_steps = int(effective_maxlenratio * enc_len / reduction_factor) if effective_maxlenratio > 0 else 1000
            # Clamp to the decoder's positional encoding limit.
            # dec_input starts at length 1 and grows by 1 each step.
            pos_limit = config.max_speech_positions - 1
            max_steps = max(min(max_steps, pos_limit), min_steps + 1, 1)
            # Disable stop token by raising threshold above sigmoid max
            eff_threshold = 1.1 if disable_stop_token else threshold

            dec_input = torch.zeros((1, 1, num_mel_bins), device=self.device)
            past_kv = None
            frames = []

            for step in range(max_steps):
                dec_hidden = self.speecht5_model.speecht5.decoder.prenet(dec_input, speaker_embedding)
                dec_out = self.speecht5_model.speecht5.decoder.wrapped_decoder(
                    hidden_states=dec_hidden[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=None,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                last_h = dec_out.last_hidden_state.squeeze(1)
                past_kv = dec_out.past_key_values

                spec = self.speecht5_model.speech_decoder_postnet.feat_out(last_h)
                spec = spec.view(1, reduction_factor, num_mel_bins)
                frames.append(spec)

                dec_input = torch.cat([dec_input, spec[:, -1:, :]], dim=1)

                if step >= min_steps:
                    # Use max over reduction_factor frames: stop when the most confident
                    # sub-frame exceeds the threshold (Tacotron-2 semantics).
                    # NOTE: sum() fires too early with reduction_factor=2 (avg threshold 0.25).
                    stop_p = torch.sigmoid(self.speecht5_model.speech_decoder_postnet.prob_out(last_h))
                    if torch.max(stop_p, dim=-1).values >= eff_threshold:
                        break

            if not frames:
                return np.zeros(16000, dtype=np.float32)

            mel_out = torch.cat(frames, dim=1)
            mel_post = self.speecht5_model.speech_decoder_postnet.postnet(mel_out)
            speech = self.vocoder(mel_post).squeeze()

        return speech.cpu().numpy()
