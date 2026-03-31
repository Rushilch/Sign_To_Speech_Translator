# VaaniVerse — Multilingual Voice & Sign Language Interpreter

VaaniVerse is a real-time desktop application that performs bidirectional translation between spoken Indian languages and Indian Sign Language (ISL). The pipeline combines MediaPipe-based pose estimation, a KNN classifier trained on hybrid hand-face feature vectors, Google Speech Recognition, and gTTS — all wired together in a Tkinter GUI with two independent operating modes.

---

## Architecture Overview

The system runs two distinct pipelines depending on mode:

**Speak → Sign** is a sequential translation pipeline: raw audio → STT → language detection → `googletrans` translation to English → hierarchical sign lookup (sentence → word → character fallback) → GIF/image playback via Pillow.

**Sign → Speech** is a real-time inference loop: webcam frames → MediaPipe Hands + Face Mesh → landmark extraction → feature normalization → KNN inference → prediction smoothing → sentence construction → `googletrans` → gTTS audio output.

---

## Feature Extraction & Model

Each frame produces a hybrid feature vector built from:

- **Hand geometry:** 21 landmarks per hand (x, y, z), normalized relative to the wrist to remove positional variance. Both hands are encoded; absent hands are zero-padded.
- **Face-relative spatial encoding:** Euclidean distances from key hand landmarks to the nose tip, providing scale-invariant upper-body context.
- **Motion placeholder:** Reserved slots in the vector for velocity/acceleration features (not yet populated — currently static-only).

The resulting vector is fed to a **distance-weighted KNN classifier** (`sklearn.neighbors.KNeighborsClassifier`, weights=`distance`). Predictions are stabilized using a fixed-length sliding window — the modal prediction over the last N frames is emitted, suppressing per-frame noise.

Training runs `build_model.py` (landmark extraction → `.npy` feature arrays + `hybrid_labels.txt`) then `train_model.py` (KNN fit → `hybrid_model.pkl` via `joblib`).

---

## Sign Lookup Hierarchy

When translating text to signs, the system queries the `ISL_CSLRT_Corpus` in order:

1. **Sentence-level:** exact phrase match against `Sentence_Level/`
2. **Word-level:** per-token lookup against `Word_Level/`
3. **Letter-level fallback:** fingerspelling from `Letter_Level/` for any unmatched token

This degrades gracefully — unknown vocabulary never hard-fails, it fingerspells instead.

---

## Getting Started

```bash
git clone https://github.com/your-username/VaaniVerse.git
cd VaaniVerse
python -m venv venv
venv\Scripts\activate

pip install opencv-python mediapipe numpy scikit-learn joblib pillow tqdm
pip install SpeechRecognition "googletrans==4.0.0-rc1" gtts

python app.py
```

> **Python 3.11 required.** MediaPipe's prebuilt wheels don't support 3.12+ cleanly as of this writing.

---

## Training Pipeline

```
data/
├── HELLO/          ← one folder per sign class
├── THANK_YOU/
└── HOW_ARE_YOU/
```

```bash
python build_model.py   # runs MediaPipe on every image, writes hybrid_data/*.npy + hybrid_labels.txt
python train_model.py   # loads .npy arrays, fits KNN, serializes to hybrid_model.pkl
```

Feature extraction is the slow step — `tqdm` progress bars track per-class processing. Re-run both scripts any time the dataset changes; the `.pkl` is not incrementally updatable.

---

## Project Structure

```
VaaniVerse/
├── app.py                  # GUI entrypoint, mode switching, camera loop
├── build_model.py          # Landmark extraction → .npy features
├── train_model.py          # KNN training → hybrid_model.pkl
├── hybrid_model.pkl        # Serialized classifier
├── hybrid_labels.txt       # Index-to-class label mapping
├── data/                   # Raw sign images (training input)
├── hybrid_data/            # Extracted feature arrays (build output)
└── images/
    └── ISL_CSLRT_Corpus/
        ├── Sentence_Level/
        ├── Word_Level/
        └── Letter_Level/
```

---

## Constraints & Known Limitations

- **Occlusion sensitivity:** Both hands and the face must be simultaneously visible. Partial occlusion produces degraded or zeroed features and reduces classification confidence significantly.
- **Static-only recognition:** The current feature vector has no temporal component. Signs distinguished primarily by motion (not hand shape) are either misclassified or collapsed into a single class.
- **KNN scalability:** Inference time scales linearly with training set size. For large vocabularies (500+ classes), consider switching to a quantized index (e.g. `faiss`) or replacing KNN with a shallow MLP.
- **`googletrans` stability:** The `4.0.0-rc1` release uses an unofficial reverse-engineered API. It can break without notice on Google's end — consider the official Cloud Translation API for production use.
- **Lighting dependence:** MediaPipe's hand detector degrades under low contrast or overexposed conditions. No preprocessing (CLAHE, histogram equalization) is currently applied.

---

## Planned Work

- **Temporal modeling:** Replace static feature vector with a sequence encoder (LSTM or TCN) operating over a rolling frame buffer to capture motion-dependent signs.
- **Confidence visualization:** Surface per-prediction KNN distance scores in the UI to flag low-confidence outputs.
- **Dataset recorder:** In-app tool to capture and label new signs directly into `data/`, streamlining dataset expansion.
- **Transformer-based recognizer:** Evaluate MediaPipe + lightweight ViT or graph attention network as a drop-in replacement for KNN.
- **Deployment:** Package as a cross-platform binary (PyInstaller) and explore a browser port via WebAssembly + TensorFlow.js.

---

## Stack

| Layer | Implementation |
|---|---|
| Language | Python 3.11 |
| UI | Tkinter |
| Vision | OpenCV, MediaPipe Hands + Face Mesh |
| ML | NumPy, Scikit-Learn (KNN), Joblib |
| Speech input | Google Speech Recognition (`SpeechRecognition`) |
| Translation | googletrans 4.0.0-rc1 |
| TTS | gTTS (Google Text-to-Speech) |
| Image rendering | Pillow |
