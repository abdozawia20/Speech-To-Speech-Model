
import os
import sys
import numpy as np

# Ensure we can import from the directory
sys.path.append(os.getcwd())

from STT_TTS_models import STT_model, TTS_model

def test_stt_dynamic_loading():
    print("\n--- Testing STT Multi-Engine Loading ---")
    stt = STT_model(engine="whisper", model_size="tiny")
    print(f"STT initialized with engine: {stt.current_engine_name}")
    assert stt.current_engine_name == "whisper"
    
    print("Switching STT to 'vosk'...")
    stt.load_engine("vosk")
    print(f"STT switched to engine: {stt.current_engine_name}")
    assert stt.current_engine_name == "vosk"

    print("Switching STT to 'google'...")
    stt.load_engine("google")
    print(f"STT switched to engine: {stt.current_engine_name}")
    
    print("Switching STT to 'assemblyai'...")
    stt.load_engine("assemblyai")
    print(f"STT switched to engine: {stt.current_engine_name}")

def test_tts_dynamic_loading():
    print("\n--- Testing TTS Dynamic Loading ---")
    tts = TTS_model(model_name="en_US-lessac-medium")
    print(f"TTS initialized with '{tts.current_model_name}'.")
    assert tts.current_model_name == "en_US-lessac-medium"

    print("Switching TTS to 'en_US-lessac-high'...")
    tts.load_model("en_US-lessac-high")
    print(f"TTS switched to '{tts.current_model_name}'.")
    assert tts.current_model_name == "en_US-lessac-high"
    
    # Verify files check
    if os.path.exists("en_US-lessac-high.onnx"):
        print("Model file 'en_US-lessac-high.onnx' verified.")
    else:
        print("Error: Model file 'en_US-lessac-high.onnx' missing.")

if __name__ == "__main__":
    try:
        test_stt_dynamic_loading()
        test_tts_dynamic_loading()
        print("\nAll dynamic loading tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
