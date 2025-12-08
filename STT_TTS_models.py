import os
import subprocess
from IPython.display import Audio, display
from faster_whisper import WhisperModel
import soundfile as sf
import librosa
import numpy as np
import torch
import ctypes
import os

# Fix for ctranslate2/faster-whisper not finding cuDNN 9 libraries
try:
    import nvidia.cudnn
    import nvidia.cublas
    
    cudnn_path = os.path.join(nvidia.cudnn.__path__[0], 'lib')
    cublas_path = os.path.join(nvidia.cublas.__path__[0], 'lib')
    
    # Load cuDNN libraries with RTLD_GLOBAL
    for lib in os.listdir(cudnn_path):
        if "libcudnn" in lib and ".so.9" in lib:
             try:
                 ctypes.CDLL(os.path.join(cudnn_path, lib), mode=ctypes.RTLD_GLOBAL)
             except Exception:
                 pass

    # Load cuBLAS libraries with RTLD_GLOBAL
    try:
         ctypes.CDLL(os.path.join(cublas_path, "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)
         ctypes.CDLL(os.path.join(cublas_path, "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

except ImportError:
    print("Warning: Could not import nvidia libraries for preloading. STT may crash on CUDA.")
except Exception as e:
    print(f"Warning: Error preloading nvidia libraries: {e}")

# 1. Download the Piper Binary
# Using the stable 2023.11.14-2 release

class TTS_model():

    def __init__(self):
        self.piper_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
        
        if self.validate():
            print("Piper binary found.")
            return
        
        print(f"Downloading Piper from: {self.piper_url}")
        subprocess.run(["wget", "-q", self.piper_url], check=True)
        subprocess.run(["tar", "-xf", "piper_linux_x86_64.tar.gz"], check=True)

        # 2. Download a Voice Model (en_US-lessac-medium)
        print("Downloading voice model...")
        subprocess.run(["wget", "-q", "-O", "voice.onnx", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"], check=True)
        subprocess.run(["wget", "-q", "-O", "voice.onnx.json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"], check=True)
        

    def validate(self):
        # 3. Locate the binary (Safety Check)
        self.piper_binary = "./piper/piper"
        if not os.path.exists(self.piper_binary):
            print("Standard path not found. Searching...")
            # Fallback search
            result = subprocess.run(["find", ".", "-name", "piper", "-type", "f", "-executable"], capture_output=True, text=True).stdout.splitlines()
            if result:
                self.piper_binary = result[0]
                print(f"Found binary at: {self.piper_binary}")
                return True
            else:
                print("Could not find the 'piper' executable after extraction.")
                return False

    def run_inference(self, text_input):
        print(f"Generating audio for the following text: \n '{text_input}'")
        cmd = f'echo "{text_input}" | "{self.piper_binary}" --model voice.onnx --output_file output.wav'
        subprocess.run(cmd, shell=True, check=True)

        # 5. Return the result
        if os.path.exists("output.wav") and os.path.getsize("output.wav") > 0:
            return Audio("output.wav")
        else:
            print("Error: output.wav was not generated or is empty.")
            return None


class STT_model():
    
    def __init__(self):
        model_size = "small"  # start with small; later you can try medium or large-v3

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        # Create model (will download on first run)
        print(f"Loading WhisperModel on {device}...")

        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading WhisperModel on {device}: {e}")
            print("Falling back to CPU int8...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
    def run_inference(self, audio_array, sample_rate):

        if audio_array is None or len(audio_array) == 0:
            raise ValueError("audio_array cannot be 'None' or empty")
        
        # Ensure audio_array is a numpy array
        if not isinstance(audio_array, (np.ndarray, list)):
             raise TypeError(f"audio_array must be a numpy array or list, got {type(audio_array)}")
        
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array)

        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        print(f"Transcribing audio file...")
        try:
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language=None
            )

            print("Detected language:", info.language)
            print("Language probability:", info.language_probability)
            print("-" * 40)

            full_text = ""

            for seg in segments:
                line = f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}"
                print(line)
                full_text += seg.text + " "

            return full_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
