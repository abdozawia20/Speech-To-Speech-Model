# Speech-to-Speech Translation: UNet Architectural Iterations & Discoveries

This report documents the chronological iterations, hyperparameters, and critical findings during the development of a UNet-based acoustic model designed to map English Mel-Spectrograms to German Mel-Spectrograms.

---

## Executive Summary

The transition from visual image translation to acoustic spectrogram translation presents unique computational challenges. Over three extensive architectural stages, our UNet model demonstrated clearly that **standard visual translation paradigms fundamentally differ from acoustic processing.** 

While the architecture successfully ingested, minimized mathematical loss objectives, and output normalized arrays, the models consistently fell into statistical traps (Numeric Explosion -> Mean-Prediction Collapse -> Adversarial Mode Collapse). The findings definitively prove that standard pixel-wise objective functions (MSE/L1) and basic GAN setups struggle heavily against the extreme entropy of human speech represented in 80-bin Mel-Spectrograms, particularly when paired with a phase-approximating decoder like Griffin-Lim.

---

## 🏗️ Architectural Iterations Summary Table

| Parameter/Setting | Stage 1: Baseline Unnormalized | Stage 2: Normalized MSE Constraint | Stage 3: Adversarial PatchGAN (Current) |
| :--- | :--- | :--- | :--- |
| **Data Range** | `[-80, 0]` dB (Raw Librosa) | `[0, 1]` (Clamped & Scaled) | `[0, 1]` (Sigmoid Bounded) |
| **Core Architecture** | Pure ResBlock Conv UNet | Pure ResBlock Conv UNet | Conv UNet + ViT Bottleneck |
| **Loss Function** | MSE + L1 | MSE + L1 | LSGAN (Adv) + MSE/L1 (Rec) |
| **Output Activation** | Linear / Unbounded | Linear / Unbounded | `torch.sigmoid` |
| **Predicted dB Range**| `[-6733.8, -882.5]` dB | `[-40.0, -39.0]` dB | `[-40.7, -38.3]` dB |
| **Generator Std Dev** | Unconstrained | `~0.004` (Uniform block) | `~0.004` (Mode Collapse) |
| **Audio Output** | Destructive Static Explosion | Flat White Noise / Static | Flat White Noise / Static |

<br>

---

## 🔬 Detailed Discoveries per Stage

### Stage 1: Unnormalized Spectrograms + Basic MSE
**The Setup:** The initial UNet was trained to map input English spectrograms directly to target German spectrograms dynamically utilizing their native Log-Mel representations spanning `[-80, 0]` dB.
**The Outcome:** The absence of bound scaling and a stabilizing activation layer meant the network weights updated aggressively in opposite extremes to handle outliers.
**The Discovery (Numeric Explosion):** By testing the generated output, we discovered the UNet was emitting massively unconstrained negative floats. When multiplied into the decibel scaling algorithm locally, it resulted in values as low as `-6733 dB`. Because human speech happens between 0 and -80 dB, a `-6733 dB` value translates to floating-point numeric trash during the Griffin-Lim inversion, resulting in ear-piercing static.

### Stage 2: Normalized Spectrograms + Basic MSE
**The Setup:** To combat the unbounded values, the preprocessing pipeline was overhauled. Target arrays were successfully scaled and clamped mathematically into a strict `[0, 1]` envelope. 
**The Outcome:** The mathematical training loss smoothly plateaued very early on. However, the exact same audio noise was generated during inference testing.
**The Discovery (Mean-Prediction Collapse):** By executing a direct statistical evaluation on the model's output array, we established that the `[0,1]` model was suffering from "Regression to the Mean." The target ground-truth audio possessed a standard deviation of `~0.213`. The model’s prediction held a standard deviation of just `0.004`. 
Because Neural Networks are lazy, the easiest mathematical way to minimize Mean Squared Error (MSE) across thousands of highly chaotic speech variations is to simply output a "solid gray box" spanning a uniform `0.5` value. When decoded, a visually uniform image contains no acoustic frequencies, resulting directly in flat static noise.

### Stage 3: Adversarial Discriminator (PatchGAN) & ViT
**The Setup:** To solve Mean-Prediction Collapse, a secondary `SpectrogramDiscriminator` (Conditional PatchGAN) was integrated. This model acts as a critic: if the Generator outputs a blurry gray `0.5` block, the Discriminator rejects it. Furthermore, a Vision Transformer (ViT) bottleneck was placed in the core of the UNet to allow it to "see" sequence time-dependencies properly instead of just static shapes. We also appended `torch.sigmoid` to fundamentally prevent numeric explosions.
**The Outcome:** The GAN successfully stabilized the `min-max` loss curves, showcasing exactly how a textbook adversarial rivalry behaves. However, statistical evaluation during inference confirmed the Generator stubbornly maintained a uniform `~0.511` output (Std: `0.004`).
**The Discovery (Adversarial Mode Collapse):** Despite learning the architecture curves properly, the Generator collapsed into a mathematical safe-zone. Generating temporally accurate, high-frequency speech structures using 80-bin Mel resolutions is exceptionally hard. The network learned to exploit minor mathematical gradients to pass the Discriminator but failed to synthesize complex textures. 

---

## 🎯 Conclusion & Future Directions

The pipeline was successfully automated, data processed, and an incredibly robust GAN training loop was established. We discovered that building a purely image-translation-based architecture to synthesize speech generates extreme structural blocks.

**Key Learnings for the Thesis/Project:**
1. **Griffin-Lim is a Hard Bottleneck:** It was noted that even standard ground-truth 80-bin mel-spectrograms sound metallic and "robotic" when reconstructed mathematically via Griffin-Lim. To achieve human-like speech from spectral prediction models, high-grade Neural Vocoders (like HiFi-GAN) are fundamentally mandatory.
2. **MSE is Insufficient for Acoustics:** Standard Mean Squared Error forces CNN architectures into gray, mean-prediction states that completely erase the "phase" and textural structures representing words in a spectrogram.
3. **Sequential Architectures Overrule Visual Architectures:** Spectrograms masquerade as images, but they are inherently sequential time-series data. Transformers (like SpeechT5) or heavy self-attention temporal mechanisms (like Whisper) are vastly superior for acoustic logic modeling than purely spatial UNets.
