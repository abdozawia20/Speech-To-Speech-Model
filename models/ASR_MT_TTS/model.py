import os
import subprocess
import io
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
from faster_whisper import WhisperModel
import vosk

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

class TTSBase:
    def load_model(self, model_name):
        pass
    def run_inference(self, text_input):
        raise NotImplementedError

PIPER_MODEL_SPECS = {}
PIPER_MODEL_SPECS["en"] = {}
PIPER_MODEL_SPECS["ar"] = {}
PIPER_MODEL_SPECS["tr"] = {}
PIPER_MODEL_SPECS["de"] = {}

PIPER_MODEL_SPECS["en"]["small"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/small/en_US-lessac-small.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/small/en_US-lessac-small.onnx.json"
}

PIPER_MODEL_SPECS["en"]["medium"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
}

PIPER_MODEL_SPECS["en"]["high"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
}

PIPER_MODEL_SPECS["ar"]["small"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx.json"
}

PIPER_MODEL_SPECS["ar"]["medium"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json"
}

PIPER_MODEL_SPECS["tr"]["medium"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/fettah/medium/tr_TR-fettah-medium.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/fettah/medium/tr_TR-fettah-medium.onnx.json"
}

PIPER_MODEL_SPECS["de"]["low"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json"
}

PIPER_MODEL_SPECS["de"]["medium"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
}

PIPER_MODEL_SPECS["de"]["high"] = {
    "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
    "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json"
}

class PiperEngine(TTSBase):
    # Define available voice models (name -> {onnx_url, json_url})

    def __init__(self, model_size="small", model_language="en"):
        self.current_model_size = None
        self.current_model_language = None
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

        self.load_model(model_size, model_language)

    def load_model(self, model_size, model_language):
        if model_size == self.current_model_size and model_language == self.current_model_language:
            print(f"Model '{model_language}':'{model_size}' is already loaded.")
            return

        if model_size not in PIPER_MODEL_SPECS[model_language]:
            print(f"Warning: Model '{model_language}':'{model_size}' not found in MODEL_SPECS. Using default 'medium'.")
            model_size = "medium"

        print(f"Loading voice model: {model_size}...")
        model_info = PIPER_MODEL_SPECS[model_language][model_size]
        
        # Define file names 
        onnx_filename = f"piper_{model_language}_{model_size}.onnx"
        json_filename = f"piper_{model_language}_{model_size}.onnx.json"

        # Check if files already exist
        if not os.path.exists(onnx_filename) or not os.path.exists(json_filename): 
            print(f"Downloading {model_info['onnx']}...")
            try:
                urllib.request.urlretrieve(model_info["onnx"], onnx_filename)
                urllib.request.urlretrieve(model_info["json"], json_filename)
            except Exception as e:
                print(f"Error downloading voice model: {e}")
                return
        else:
             print(f"Model files for {model_info['onnx']} already exist.")

        self.current_model_name = model_info['onnx']
        self.current_onnx_path = onnx_filename
        print(f"Voice model '{model_info['onnx']}' ready.")
        

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

        cmd = [self.piper_binary, "--model", self.current_onnx_path, "--output_file", "-"]
        
        try:
           # Use subprocess.run to pipe input and capture output directly
           result = subprocess.run(
               cmd,
               input=text_input.encode('utf-8'),
               capture_output=True,
               check=True
           )
           
           # Read from memory
           with io.BytesIO(result.stdout) as wav_io:
               data, samplerate = sf.read(wav_io)
               
           return {
               'audio': {
                   'array': data,
                   'sampling_rate': samplerate
               }
           }
        except subprocess.CalledProcessError as e:
            print(f"Error running Piper: {e}")
            if e.stderr:
                print(f"Stderr: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"Error processing audio output: {e}")
            return None

class SystemEngine(TTSBase):
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

class ElevenLabsEngine(TTSBase):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            print("Warning: No ElevenLabs API key provided.")

    def load_model(self, model_name):
        print(f"ElevenLabs model '{model_name}' selected (Mock).")

    def run_inference(self, text_input):
        print("ElevenLabs inference requires valid API Key. (This is a placeholder)")
        return None

class OpenAIEngine(TTSBase):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
             print("Warning: No OpenAI API key provided.")
    
    def load_model(self, model_name):
        print(f"OpenAI TTS model '{model_name}' selected (Mock).")

    def run_inference(self, text_input):
        print("OpenAI inference requires valid API Key. (This is a placeholder)")
        return None

class TTSEngine():
    """
    Facade for switching between TTS engines.
    Engines: 'piper', 'system', 'elevenlabs', 'openai'
    """
    def __init__(self, engine="piper", model_size="small", language="en"):
        self.engine = None
        self.current_engine_name = None

        engine = engine.lower()
        print(f"Switching TTS Engine to: {engine}...")
        
        if engine == "piper":
            self.engine = PiperEngine(model_size, language)
        elif engine == "system":
            self.engine = SystemEngine(model_name)
        elif engine == "elevenlabs":
            # self.engine = ElevenLabsEngine(api_key)
            raise NotImplementedError
        elif engine == "openai":
            # self.engine = OpenAIEngine(api_key)
            raise NotImplementedError
        else:
            print(f"WARNING: Engine settings not found for '{engine}:{model_size}-{language}'. Defaulting to Piper:en-small.")
            self.__init__(engine="piper", model_size="small", language="en")
        
        self.current_engine_name = engine

    def run_inference(self, text_input):
        if self.engine:
            return self.engine.run_inference(text_input)
        return None



class STTBase:
    # This implies that the user wants a model that isn't present in the library
    def load_model(self, model_name, model_size="small"):
        raise NotImplementedError
    def transcribe(self, audio_array, sample_rate):
        raise NotImplementedError

class WhisperEngine(STTBase):
    """
    Whisper STT Engine
    This engine uses OpenAI's Whisper for speech-to-text transcription. 
    This model is used only for comparing the performance of the main model.
    External Documentation: https://github.com/openai/whisper

    Supported Models (languages are autodetected based on the audio):
    - small
    - medium
    - large

    """


    def __init__(self, model_size, language):
        self.model = None
        self.language = language

    def load_model(self, model_size):

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
        
        return self.model

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
                language=self.language
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

VOSK_MODEL_SPECS = {}
VOSK_MODEL_SPECS["en"] = {}
VOSK_MODEL_SPECS["ar"] = {}
VOSK_MODEL_SPECS["tr"] = {}
VOSK_MODEL_SPECS["de"] = {}

VOSK_MODEL_SPECS["en"]["small"] = "vosk-model-small-en-us-0.15"
VOSK_MODEL_SPECS["en"]["large"] = "vosk-model-en-us-0.22"

VOSK_MODEL_SPECS["ar"]["small"] = "vosk-model-ar-mgb2-0.4"
VOSK_MODEL_SPECS["ar"]["large"] = "vosk-model-ar-0.22-linto-1.1.0"

VOSK_MODEL_SPECS["tr"]["small"] = "vosk-model-small-tr-0.3"

VOSK_MODEL_SPECS["de"]["small"] = "vosk-model-small-de-0.15"

class VoskEngine(STTBase):
    """
    Vosk STT Engine
    This engine uses Vosk for speech-to-text transcription. 
    This model is used only for comparing the performance of the main model.
    External Documentation: https://alphacephei.com/vosk/models
    
    Current supported models:
    en: small, large
    ar: small, large
    tr: small
    de: small
    """
    
    def __init__(self, model_size, language):
        self.model = None
        self.load_model(model_size, language)

    def load_model(self, model_size, language):
        model_path = VOSK_MODEL_SPECS[language][model_size]
        # Simple auto-download
        if not os.path.exists(model_path):
            print(f"Vosk model '{model_path}' not found, attempting to download...")
            url = f"https://alphacephei.com/vosk/models/{model_path}.zip"
            print(f"Downloading {model_path} from {url}...")
            zip_name = f"{model_path}.zip"
            try:
                urllib.request.urlretrieve(url, zip_name)
                print("Extracting model...")
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    zip_ref.extractall()
                # Determine extraction folder name (sometimes it extracts to a folder inside)
                # For this zip, it extracts to "vosk-model-small-en-us-0.15"
            except Exception as e:
                print(f"Error downloading vosk model: {e}")
                return vosk.Model(model_path)

        if os.path.exists(model_path):
            print(f"Loading Vosk model '{model_path}'...")
            vosk.SetLogLevel(-1) # Silence generic logs
            self.model = vosk.Model(model_path)
            return self.model
        else:
            raise FileNotFoundError(f"Error: Vosk model path '{model_path}' does not exist.")
        

    def transcribe(self, audio_array, sample_rate):
        if self.model is None:
            raise ValueError("Vosk model not loaded.")

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

class GoogleEngine(STTBase):
    def __init__(self):
        print("Google Speech Engine initialized (Placeholder). Needs API Key.")
    def load_model(self, model_name):
        pass
    def transcribe(self, audio_array, sample_rate):
        return "[Google Speech Placeholder] Transcription requires API Key."

class AssemblyAIEngine(STTBase):
    def __init__(self):
        print("AssemblyAI Engine initialized (Placeholder). Needs API Key.")
    def load_model(self, model_name):
        pass
    def transcribe(self, audio_array, sample_rate):
        return "[AssemblyAI Placeholder] Transcription requires API Key."

class STTEngine():
    """
    Facade for swtiching between STT engines.
    Engines: 'whisper', 'vosk', 'google', 'assemblyai'
    """
    def __init__(self, engine="whisper", language="en", model_size="small"):
        self.engine_name = engine
        self.language = language
        self.model_size = model_size

        engine = engine.lower()
        print(f"Switching STT Engine to: {engine} in the ({self.language}) language...")
        
        if engine == "whisper":
            self.engine = WhisperEngine(model_size=self.model_size, language=self.language)
            self.engine.load_model(model_size=self.model_size)
        elif engine == "vosk":
            self.engine = VoskEngine(model_size=self.model_size, language=self.language)
            self.engine.load_model(model_size=self.model_size, language=self.language)
        elif engine == "google":
            # self.engine = GoogleEngine()
            raise NotImplementedError
        elif engine == "assemblyai":
            # self.engine = AssemblyAIEngine()
            raise NotImplementedError
        else:
            print(f"WARNING: Engine settings not found for '{engine}:{model_size}-{language}'. Defaulting to Whisper:en-small.")
            self.__init__(engine="whisper", language="en", model_size="small")

    def transcribe(self, audio_array, sample_rate):
        if self.engine:
            return self.engine.transcribe(audio_array, sample_rate)
        return "Error: No engine loaded."
