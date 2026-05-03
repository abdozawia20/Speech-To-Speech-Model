# 1. Start with an official PyTorch image loaded with CUDA (GPU support)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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

# 7. Copy the models folder and all files in the root directory (excluding other folders)
COPY *.py *.txt *.yml *.md ./
COPY models/ ./models/
