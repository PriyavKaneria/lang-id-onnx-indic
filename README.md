# NLTM-LID-ONNX: Edge-Ready Spoken Language Identification

**A lightweight, ONNX-ported version of the NLTM-LID system for on-device Indian Language Identification.**

---

### ⚠️ Attribution & Acknowledgements
This repository is an **inference-only optimization** of the original work developed by **IIT Mandi and IIT Dharwad** under the **Natural Language Translation Mission (NLTM)**.

- **Original Repository:** [NLTM-LID/LID-version-2.0](https://github.com/NLTM-LID/LID-version-2.0)
- **Original Paper/Project:** "Speech Technologies in Indian Languages"
- **Model Architecture:** ccc-wav2vec (Foundation Model) + LSTM-Attention (Classifier)

This repository does **not** claim ownership of the underlying model weights or research. It provides **ONNX (Open Neural Network Exchange)** versions of these models to enable deployment on mobile devices (Android/iOS), IoT, and web browsers.

---

### Supported Languages
1. Assamese (`asm`)
2. Bengali (`ben`)
3. Gujarati (`guj`)
4. Hindi (`hin`)
5. Kannada (`kan`)
6. Malayalam (`mal`)
7. Marathi (`mar`)
8. Odia (`odi`)
9. Punjabi (`pun`)
10. Tamil (`tam`)
11. Telugu (`tel`)
12. English (`eng`)

### Models Included

The original PyTorch models (~1.2GB) have been converted to ONNX and quantized to 8-bit integers (Int8) to reduce size by **75%** while maintaining ~99% confidence.

| Model File | Size | Description |
| :--- | :--- | :--- |
| **`ccc_wav2vec_quantized.onnx`** | ~320 MB | **Feature Extractor.** Takes raw audio (16kHz) and outputs high-dimensional embeddings. Based on `wav2vec2-large-xlsr-53`. |
| **`lid_classifier_quantized.onnx`** | ~5 MB | **Classifier.** Takes embeddings from the extractor and outputs language probabilities. Uses Bi-LSTM + Attention. |

### Technical Details for Mobile Implementation

If you are porting this to **Android (Kotlin)** or **iOS (Swift)**, follow this pipeline logic:

1.  **Input Audio:**
    * Format: Mono (1 channel)
    * Sample Rate: 16,000 Hz (16kHz)
    * Duration: Min 1.5s, Max ~15s (recommended).

2.  **Step 1: Feature Extraction (Wav2Vec)**
    * **Input Name:** `input_values`
    * **Input Shape:** `[1, Time]` (Float32 array of audio samples).
    * **Output Name:** `hidden_features`
    * **Output Shape:** `[1, Frames, 1024]`

3.  **Step 2: Classification (LSTM)**
    The classifier expects two "views" of the features (High Res and Low Res). You must perform this reshaping logic in Kotlin/Swift:
    
    * **Preprocessing:** Normalize the features (Mean=0, Std=1).
    * **Input 1 (`x1`):** Sliding window (Length 20, Stride 1).
    * **Input 2 (`x2`):** Sliding window (Length 50, Stride 3).
    * **Transpose:** Swap axes to `[Batch, Sequence, Feature]` before inference.

### Quick Start (Python)

To test the models on your desktop before deploying to mobile:

1.  **Install dependencies:**
    ```bash
    pip install onnxruntime numpy torch torchaudio
    ```

2.  **Run Inference:**
    ```bash
    python inference.py path/to/your_audio.wav
    ```

### License

The original work is licensed under the **NLTM License**. Please refer to the [original repository](https://github.com/NLTM-LID/LID-version-2.0) for specific licensing terms regarding commercial usage and redistribution.

The code in *this* repository (extraction and inference scripts) is provided for educational and interoperability purposes.