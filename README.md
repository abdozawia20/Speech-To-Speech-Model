# Speech-To-Speech Model

This project implements a Speech-To-Speech model pipeline, featuring components for dataset loading, audio encoding, and speech processing.

## Features

-   **Dataset Loading**:
    -   Supports loading and harmonizing datasets from **Fleurs** (English, Arabic, Turkish) and **Voxpopuli** (English).
    -   Includes data transformation and concatenation logic to create unified datasets for training.

-   **Audio Encoders**:
    -   **Spectrogram Encoder**: Converts audio arrays to dB-scaled spectrograms using `librosa`.
    -   **Wav2VecEncoder**: Utilizes the `facebook/wav2vec2-base-960h` model to extract feature vectors from audio.
    -   **VQGANEncoder**: Uses the `facebook/encodec_24khz` model for encoding audio into discrete codes and decoding them back to audio.

-   **Model Components**:
    -   **STT (Speech-to-Text)**: Components for converting speech to text (see `STT model.ipynb`).
    -   **TTS (Text-to-Speech)**: Components for converting text to speech (see `TTS model.ipynb`).
    -   **Orchestration**: `main.ipynb` serves as the main entry point for running and testing the pipeline.

## Installation

To run this project, you need to install the following dependencies:

```bash
pip install datasets==3.6.0 librosa torch matplotlib transformers
```

## Usage

1.  **Data Loading**: Use `dataset_loader.py` to load and preprocess the training data.
    ```python
    from dataset_loader import load_train_data
    data = load_train_data()
    ```

2.  **Encoding**: Use `encoders.py` to initialize and use the encoders.
    ```python
    from encoders import Wav2VecEncoder, VQGANEncoder
    
    wav2vec = Wav2VecEncoder()
    vqgan = VQGANEncoder()
    ```

3.  **Running the Notebooks**: Open `main.ipynb`, `STT model.ipynb`, or `TTS model.ipynb` in Jupyter Notebook or Google Colab to explore the models and training procedures.

## Project Structure

-   `dataset_loader.py`: Script for loading and transforming datasets.
-   `encoders.py`: Definitions for various audio encoders.
-   `main.ipynb`: Main notebook for orchestration and testing.
-   `STT model.ipynb`: Notebook focused on the Speech-to-Text model.
-   `TTS model.ipynb`: Notebook focused on the Text-to-Speech model.
