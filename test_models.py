
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
    print("\n--- Testing TTS Multi-Engine Loading ---")
    tts = TTS_model(engine="piper", model_name="en_US-lessac-medium")
    print(f"TTS initialized with engine: {tts.current_engine_name}")
    assert tts.current_engine_name == "piper"

    print("Switching TTS engine to 'system' (pyttsx3)...")
    tts.load_engine("system")
    print(f"TTS switched to engine: {tts.current_engine_name}")
    assert tts.current_engine_name == "system"
    
    # Test dummy inference (optional, might make sound)
    # print("Testing System TTS generation...")
    # tts.run_inference("Hello from System TTS")

    print("Switching TTS engine back to 'piper'...")
    tts.load_engine("piper", model_name="en_US-lessac-high")
    print(f"TTS switched to engine: {tts.current_engine_name}")
    assert tts.current_engine_name == "piper"
    
    # Verify Piper model file exists
    if os.path.exists("en_US-lessac-high.onnx"):
        print("Piper model file 'en_US-lessac-high.onnx' verified.")
    else:
        print("Error: Piper model file missing.")

if __name__ == "__main__":
    try:
        test_stt_dynamic_loading()
        test_tts_dynamic_loading()
        print("\nAll dynamic loading tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
