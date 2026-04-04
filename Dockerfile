# 1. Start with an official PyTorch image loaded with CUDA (GPU support)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 2. Prevent interactive prompts from blocking the build (e.g., asking for timezones)
ENV DEBIAN_FRONTEND=noninteractive

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install essential OS-level audio packages
# - libsndfile1 & ffmpeg: Required by librosa and soundfile to read/write audio
# - espeak: Required by pyttsx3 for offline text-to-speech
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements file into the container
COPY requirements.txt .

# 6. Install your specific Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your actual project code into the container
COPY . .

# 8. Choose which model pipeline to run.
# Uncomment the CMD for the model you want to execute when the container starts.
# Alternatively, you can override this safely using `docker run ... python path/to/script.py`

# --- Baseline (Cascaded ASR -> MT -> TTS) ---
CMD ["python", "models/ASR_MT_TTS/analysis_pipeline.py"]

# --- SpeechT5 Model ---
# CMD ["python", "models/SpeechT5/evaluate_speecht5.py"]

# --- UNet Model ---
# CMD ["python", "models/UNet/model.py"]

# To build this container image, run:
# docker build -t speech-to-speech:latest .
