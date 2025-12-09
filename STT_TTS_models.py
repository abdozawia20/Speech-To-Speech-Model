import os
import subprocess
from IPython.display import Audio, display
import soundfile as sf
import getpass
import librosa
import numpy as np
import torch
import ctypes
import urllib.request
import tarfile
import platform
import zipfile
import json

# AMD Fix: Force MKL to use AVX2 on AMD CPUs for better performance
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
# Allow duplicate OpenMP libraries (common fix for some environments)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix for ctranslate2/faster-whisper not finding cuDNN 9 libraries
if platform.system() != "Windows":
    try:
        import nvidia.cudnn
        import nvidia.cublas
        
        cudnn_path = os.path.join(nvidia.cudnn.__path__[0], 'lib')
        cublas_path = os.path.join(nvidia.cublas.__path__[0], 'lib')
        
        # Load cuDNN libraries with RTLD_GLOBAL
        if os.path.exists(cudnn_path):
            for lib in os.listdir(cudnn_path):
                if "libcudnn" in lib and ".so.9" in lib:
                     try:
                         ctypes.CDLL(os.path.join(cudnn_path, lib), mode=ctypes.RTLD_GLOBAL)
                     except Exception:
                         pass

        # Load cuBLAS libraries with RTLD_GLOBAL
        if os.path.exists(cublas_path):
            try:
                 ctypes.CDLL(os.path.join(cublas_path, "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)
                 ctypes.CDLL(os.path.join(cublas_path, "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

    except ImportError:
        pass # Optional nvidia libs not installed
    except Exception as e:
        print(f"Warning: Error preloading nvidia libraries: {e}")

# 1. Download the Piper Binary
# Using the stable 2023.11.14-2 release

# Base Interface
class TTSEngine:
    def load_model(self, model_name):
        pass
    def run_inference(self, text_input):
        raise NotImplementedError

# Concrete Implementation: Piper (Offline, High Quality)
class PiperEngine(TTSEngine):
    # Define available voice models (name -> {onnx_url, json_url})
    VOICE_MODELS = {
        "en_US-lessac-medium": {
            "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        },
        "en_US-lessac-high": {
            "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
            "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
        },
         "en_US-libritts-high": {
            "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx",
            "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json"
        }
    }

    def __init__(self, model_name="en_US-lessac-medium"):
        self.current_model_name = None
        self.piper_binary = None
        self.piper_url = None
        
        # OS Detection
        self.system = platform.system()
        if self.system == "Windows":
             self.piper_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
             self.archive_name = "piper_windows_amd64.zip"
             self.executable_name = "piper.exe"
        else:
             self.piper_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
             self.archive_name = "piper_linux_x86_64.tar.gz"
             self.executable_name = "piper"

        if self.validate():
            print("Piper binary found.")
        else:
            print(f"Downloading Piper from: {self.piper_url}")
            try:
                urllib.request.urlretrieve(self.piper_url, self.archive_name)
                print("Extracting Piper...")
                if self.archive_name.endswith(".zip"):
                    with zipfile.ZipFile(self.archive_name, 'r') as zip_ref:
                        zip_ref.extractall()
                else:
                    with tarfile.open(self.archive_name, "r:gz") as tar:
                        tar.extractall()
                self.validate() # Re-validate to find the binary
            except Exception as e:
                print(f"Error downloading/extracting Piper: {e}")

        self.load_model(model_name)

    def load_model(self, model_name):
        if model_name == self.current_model_name:
            print(f"Model '{model_name}' is already loaded.")
            return

        if model_name not in self.VOICE_MODELS:
            print(f"Warning: Model '{model_name}' not found in VOICE_MODELS. Using default 'en_US-lessac-medium'.")
            model_name = "en_US-lessac-medium"

        print(f"Loading voice model: {model_name}...")
        model_info = self.VOICE_MODELS[model_name]
        
        # Define file names 
        onnx_filename = f"{model_name}.onnx"
        json_filename = f"{model_name}.onnx.json"

        # Check if files already exist
        if not os.path.exists(onnx_filename) or not os.path.exists(json_filename): 
            print(f"Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(model_info["onnx"], onnx_filename)
                urllib.request.urlretrieve(model_info["json"], json_filename)
            except Exception as e:
                print(f"Error downloading voice model: {e}")
                return
        else:
             print(f"Model files for {model_name} already exist.")

        self.current_model_name = model_name
        self.current_onnx_path = onnx_filename
        print(f"Voice model '{model_name}' ready.")
        

    def validate(self):
        # Locate the binary
        self.piper_binary = f"./piper/{self.executable_name}"
        if not os.path.exists(self.piper_binary):
            print("Standard path not found. Searching...")
            # Fallback search
            try: 
                result = subprocess.run(["find", ".", "-name", self.executable_name, "-type", "f", "-executable"], capture_output=True, text=True) if self.system != "Windows" else subprocess.run(["dir", "/s", "/b", self.executable_name], shell=True, capture_output=True, text=True)
                output = result.stdout.strip().splitlines()
                if output:
                     self.piper_binary = output[0]
                     print(f"Found binary at: {self.piper_binary}")
                     return True
            except Exception:
                pass
                
            print(f"Could not find the '{self.executable_name}' executable after extraction.")
            return False
        return True

    def run_inference(self, text_input):
        print(f"Generating audio for the following text (Piper): \n '{text_input}'")
        if not self.current_onnx_path:
             print("Error: No voice model loaded.")
             return None

        cmd = f'echo "{text_input}" | "{self.piper_binary}" --model {self.current_onnx_path} --output_file output.wav'
        try:
           # Windows Popen fix for pipes
           if self.system == "Windows":
               subprocess.run(cmd, shell=True, check=True)
           else:
               subprocess.run(cmd, shell=True, check=True)
           
           return "output.wav"
        except subprocess.CalledProcessError as e:
            print(f"Error running Piper: {e}")
            return None

# Concrete Implementation: System (pyttsx3)
class SystemEngine(TTSEngine):
    def __init__(self, voice_id=None):
        import pyttsx3 # Lazy import
        try:
            self.engine = pyttsx3.init()
            if voice_id:
                self.load_model(voice_id)
        except Exception as e:
            print(f"Error initializing system TTS: {e}")
            self.engine = None

    def load_model(self, model_name):
        # model_name here refers to voice ID or index
        if not self.engine: return
        
        voices = self.engine.getProperty('voices')
        # Try to match by ID or Name
        found = False
        for v in voices:
            if model_name in v.id or model_name in v.name:
                self.engine.setProperty('voice', v.id)
                print(f"System TTS Voice set to: {v.name}")
                found = True
                break
        if not found:
            print(f"Voice '{model_name}' not found. Using default.")

    def run_inference(self, text_input):
        import pyttsx3
        if not self.engine:
             return None
        
        print(f"Generating audio for the following text (System): \n '{text_input}'")
        output_file = "output.wav"
        try:
            self.engine.save_to_file(text_input, output_file)
            self.engine.runAndWait()
            return output_file
        except Exception as e:
            print(f"Error running System TTS: {e}")
            return None

# Concrete Implementation: ElevenLabs (Placeholder)
class ElevenLabsEngine(TTSEngine):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            print("Warning: No ElevenLabs API key provided.")

    def load_model(self, model_name):
        print(f"ElevenLabs model '{model_name}' selected (Mock).")

    def run_inference(self, text_input):
        print("ElevenLabs inference requires valid API Key. (This is a placeholder)")
        return None

# Concrete Implementation: OpenAI (Placeholder)
class OpenAIEngine(TTSEngine):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
             print("Warning: No OpenAI API key provided.")
    
    def load_model(self, model_name):
        print(f"OpenAI TTS model '{model_name}' selected (Mock).")

    def run_inference(self, text_input):
        print("OpenAI inference requires valid API Key. (This is a placeholder)")
        return None

# Facade for switching between TTS engines
class TTS_model():
    """
    Facade for switching between TTS engines.
    Engines: 'piper', 'system', 'elevenlabs', 'openai'
    """
    def __init__(self, engine="piper", **kwargs):
        self.engine = None
        self.current_engine_name = None
        self.load_engine(engine, **kwargs)

    def load_engine(self, engine_name, **kwargs):
        engine_name = engine_name.lower()
        print(f"Switching TTS Engine to: {engine_name}...")
        
        if engine_name == "piper":
            model_name = kwargs.get("model_name", "en_US-lessac-medium")
            self.engine = PiperEngine(model_name)
        elif engine_name == "system":
            self.engine = SystemEngine(kwargs.get("model_name"))
        elif engine_name == "elevenlabs":
            self.engine = ElevenLabsEngine(kwargs.get("api_key"))
        elif engine_name == "openai":
            self.engine = OpenAIEngine(kwargs.get("api_key"))
        else:
            print(f"Unknown engine '{engine_name}'. Defaulting to Piper.")
            self.engine = PiperEngine()
            engine_name = "piper"
        
        self.current_engine_name = engine_name

    def load_model(self, model_name):
        if self.engine:
            self.engine.load_model(model_name)

    def run_inference(self, text_input):
        if self.engine:
            return self.engine.run_inference(text_input)
        return None



class STTEngine:
    def load_model(self, model_name):
        raise NotImplementedError
    def transcribe(self, audio_array, sample_rate):
        raise NotImplementedError

class WhisperEngine(STTEngine):
    def __init__(self, model_size="small"):
        self.model = None
        self.load_model(model_size)

    def load_model(self, model_size):
        from faster_whisper import WhisperModel # Lazy import

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        print(f"Loading WhisperModel ({model_size}) on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading WhisperModel on {device}: {e}")
            print("Falling back to CPU int8...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_array, sample_rate):
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
        
        print(f"Transcribing audio file (Whisper)...")
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

class VoskEngine(STTEngine):
    def __init__(self, model_name="vosk-model-small-en-us-0.15"):
        self.model = None
        self.rec = None
        self.load_model(model_name)

    def load_model(self, model_name):
        import vosk # Lazy import

        model_path = model_name
        # Simple auto-download for the specific small English model
        if not os.path.exists(model_path):
            print(f"Vosk model '{model_name}' not found.")
            if model_name == "vosk-model-small-en-us-0.15":
                url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                print(f"Downloading {model_name} from {url}...")
                zip_name = f"{model_name}.zip"
                try:
                    urllib.request.urlretrieve(url, zip_name)
                    print("Extracting model...")
                    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                        zip_ref.extractall()
                    # Determine extraction folder name (sometimes it extracts to a folder inside)
                    # For this zip, it extracts to "vosk-model-small-en-us-0.15"
                except Exception as e:
                    print(f"Error downloading vosk model: {e}")
                    return

        if os.path.exists(model_path):
            print(f"Loading Vosk model '{model_name}'...")
            vosk.SetLogLevel(-1) # Silence generic logs
            self.model = vosk.Model(model_path)
            # KaldiRecognizer doesn't need init here per se, but useful to have
        else:
            print(f"Error: Vosk model path '{model_path}' does not exist.")

    def transcribe(self, audio_array, sample_rate):
        import vosk # Lazy import
        if self.model is None:
            return "Error: Vosk model not loaded."

        # Vosk expects 16kHz PCM 16-bit mono
        if sample_rate != 16000:
             audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # Convert float32 numpy array to int16 bytes
        # Audio is usually normalized -1.0 to 1.0. Scale to 32767
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        rec = vosk.KaldiRecognizer(self.model, 16000)
        rec.AcceptWaveform(audio_bytes)
        res = json.loads(rec.FinalResult())
        return res.get("text", "")

class GoogleEngine(STTEngine):
    def __init__(self):
        print("Google Speech Engine initialized (Placeholder). Needs API Key.")
    def load_model(self, model_name):
        pass
    def transcribe(self, audio_array, sample_rate):
        return "[Google Speech Placeholder] Transcription requires API Key."

class AssemblyAIEngine(STTEngine):
    def __init__(self):
        print("AssemblyAI Engine initialized (Placeholder). Needs API Key.")
    def load_model(self, model_name):
        pass
    def transcribe(self, audio_array, sample_rate):
        return "[AssemblyAI Placeholder] Transcription requires API Key."

class STT_model():
    """
    Facade for swtiching between STT engines.
    Engines: 'whisper', 'vosk', 'google', 'assemblyai'
    """
    def __init__(self, engine="whisper", **kwargs):
        self.engine = None
        self.current_engine_name = None
        self.load_engine(engine, **kwargs)

    def load_engine(self, engine_name, **kwargs):
        engine_name = engine_name.lower()
        print(f"Switching STT Engine to: {engine_name}...")
        
        if engine_name == "whisper":
            model_size = kwargs.get("model_size", "small")
            self.engine = WhisperEngine(model_size)
        elif engine_name == "vosk":
            model_name = kwargs.get("model_name", "vosk-model-small-en-us-0.15")
            self.engine = VoskEngine(model_name)
        elif engine_name == "google":
            self.engine = GoogleEngine()
        elif engine_name == "assemblyai":
            self.engine = AssemblyAIEngine()
        else:
            print(f"Unknown engine '{engine_name}'. Defaulting to Whisper.")
            self.engine = WhisperEngine()
            engine_name = "whisper"
        
        self.current_engine_name = engine_name

    def load_model(self, model_name):
        # Delegate specific model loading to the engine if supported
        if self.engine:
            self.engine.load_model(model_name)

    def run_inference(self, audio_array, sample_rate):
        if self.engine:
            return self.engine.transcribe(audio_array, sample_rate)
        return "Error: No engine loaded."
