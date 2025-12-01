import onnxruntime as ort
import numpy as np
import torch
import torchaudio
import argparse
import os

# =========================================================
# 1. CONFIGURATION
# =========================================================
WAV2VEC_MODEL = "ccc_wav2vec_quantized.onnx"
CLASSIFIER_MODEL = "lid_classifier_quantized.onnx"

# Language Mapping
ID2LANG = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 
           6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}

# Feature Extraction Params
TARGET_SAMPLE_RATE = 16000
LOOK_BACK1 = 20 
LOOK_BACK2 = 50

# =========================================================
# 2. AUDIO PRE-PROCESSING
# =========================================================
def load_and_preprocess_audio(path):
    print(f"Loading {path}...")
    speech_array, sampling_rate = torchaudio.load(path)
    
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    
    if sampling_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sampling_rate, TARGET_SAMPLE_RATE)
        speech_array = resampler(speech_array)
    
    speech_array = speech_array.squeeze(0)
    return speech_array.numpy()

# =========================================================
# 3. MODEL INFERENCE
# =========================================================

def run_inference(audio_path):
    # --- Step A: Wav2Vec Feature Extraction ---
    audio_data = load_and_preprocess_audio(audio_path)
    
    # Prepare Input
    input_values = audio_data[np.newaxis, :] 
    attention_mask = np.ones_like(input_values, dtype=np.int64)

    # Start Session
    ort_sess_w2v = ort.InferenceSession(WAV2VEC_MODEL)
    
    # DYNAMIC INPUT HANDLING (The Fix)
    w2v_inputs = {}
    model_inputs = ort_sess_w2v.get_inputs()
    
    # 1. Add Audio Input
    w2v_inputs[model_inputs[0].name] = input_values
    
    # 2. Add Mask Input (Only if the model asks for it)
    if len(model_inputs) > 1:
        w2v_inputs[model_inputs[1].name] = attention_mask
        print("Model uses Attention Mask.")
    else:
        print("Model optimized Attention Mask away (using Audio only).")

    # Run Wav2Vec
    print("Running Wav2Vec (ONNX)...")
    w2v_outputs = ort_sess_w2v.run(None, w2v_inputs)
    hidden_features = w2v_outputs[0] 
    
    # --- Step B: LSTM Input Preparation ---
    X = hidden_features[0] 
    
    # Normalize
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X = (X - mu) / std
    
    # High Res (Lookback 1)
    Xdata1 = []
    for i in range(0, len(X) - LOOK_BACK1, 1):
        a = X[i:(i + LOOK_BACK1), :]
        Xdata1.append(a)
    Xdata1 = np.array(Xdata1)
    
    # Low Res (Lookback 2)
    Xdata2 = []
    for i in range(0, len(X) - LOOK_BACK2, 2):
        b = X[i + 1:(i + LOOK_BACK2):3, :]
        Xdata2.append(b)
    Xdata2 = np.array(Xdata2)
    
    # Check if audio was too short
    if len(Xdata1) == 0 or len(Xdata2) == 0:
        print("Error: Audio too short for analysis.")
        return

    # Swap axes for LSTM: (Lookback, NumWindows, Features)
    X1 = np.swapaxes(Xdata1, 0, 1).astype(np.float32)
    X2 = np.swapaxes(Xdata2, 0, 1).astype(np.float32)
    
    # --- Step C: Classifier Inference ---
    ort_sess_clf = ort.InferenceSession(CLASSIFIER_MODEL)
    
    print("Running Classifier (ONNX)...")
    clf_inputs = {
        ort_sess_clf.get_inputs()[0].name: X1,
        ort_sess_clf.get_inputs()[1].name: X2
    }
    
    clf_outputs = ort_sess_clf.run(None, clf_inputs)
    lang_output = clf_outputs[0]
    
    # --- Step D: Post-Processing ---
    output = lang_output[0]
    pred_probs = np.exp(output) / np.sum(np.exp(output))
    pred_idx = np.argmax(pred_probs)
    
    predicted_lang = ID2LANG[pred_idx]
    confidence = pred_probs[pred_idx]
    
    print("\n" + "=" * 30)
    print(f"PREDICTION: {predicted_lang.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("=" * 30)
    print("Top 3 Probabilities:")
    
    # Sort and show top 3
    sorted_indices = np.argsort(pred_probs)[::-1]
    for i in range(3):
        idx = sorted_indices[i]
        print(f"{ID2LANG[idx].upper()}: {pred_probs[idx]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to WAV file")
    args = parser.parse_args()
    
    if os.path.exists(args.audio_path):
        run_inference(args.audio_path)
    else:
        print("File not found.")