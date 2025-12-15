import torch
import numpy as np
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor, SpeechT5HifiGan

class SpeechBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained components
        print("Loading SpeechT5 components...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

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
        
        # Check if input has 'input_values' or 'input_features' depending on processor specifics
        # The SpeechT5Processor usually returns input_values for audio
        
        # Generate dummy speaker embedding (1, 512)
        # SpeechT5 requires speaker embeddings for generating speech
        speaker_embeddings = torch.randn((1, 512))

        # Generate speech
        # We pass the input_values from the processor
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_values"], 
                speaker_embeddings, 
                vocoder=self.vocoder
            )

        return speech.numpy()

if __name__ == "__main__":
    print("Initializing SpeechBaseline...")
    baseline = SpeechBaseline()
    
    # Create a dummy audio signal (e.g., 1 second of silence/noise at 16kHz)
    dummy_sr = 16000
    dummy_audio = np.random.uniform(-0.5, 0.5, size=(16000,))
    
    print(f"Running prediction on dummy audio shape: {dummy_audio.shape}")
    try:
        output_audio = baseline.predict(dummy_audio, dummy_sr)
        print("Prediction successful.")
        print(f"Output audio shape: {output_audio.shape}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
