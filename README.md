---

# VaaniVerse — Multilingual Voice & Sign Language Interpreter

VaaniVerse is a desktop app that lets hearing and hearing-impaired people communicate in real time. It converts spoken Indian languages into Indian Sign Language (ISL) visuals, and reads ISL signs back as speech — bridging a gap that most software doesn't even try to address.

---

## What It Does

### Speak → Sign

You speak in any supported Indian language. The app translates your words to English, then looks up the matching sign — first as a full sentence, then word by word, and finally letter by letter if nothing else fits. The result plays back as images or GIFs on screen.

### Sign → Speech

You sign in front of your webcam. MediaPipe tracks your hands and face in real time, extracting landmark positions that a KNN classifier uses to identify the sign. Recognized signs get stitched into a sentence, translated into your language of choice, and read aloud via Google TTS.

### Language Support

English, Hindi, Telugu, Tamil, Bengali, Marathi, Gujarati, Kannada.

---

## How It Works Under the Hood

**Computer vision:** MediaPipe detects 21 hand landmarks per hand, plus a nose reference from face detection. Positions are normalized and encoded into a spatial feature vector.

**Recognition:** A K-Nearest Neighbors classifier, trained on a hybrid feature set, matches incoming vectors to known sign classes. Predictions are smoothed over a short window to avoid jitter.

**Speech pipeline:** Google Speech Recognition handles mic input. `googletrans` handles translation. `gTTS` handles voice output.

---

## Getting Started

```bash
git clone https://github.com/your-username/VaaniVerse.git
cd VaaniVerse
python -m venv venv
venv\Scripts\activate
pip install opencv-python mediapipe numpy scikit-learn joblib pillow tqdm
pip install SpeechRecognition googletrans==4.0.0-rc1 gtts
python app.py
```

> Python 3.11 is strongly recommended — MediaPipe has known issues on newer versions.

---

## Training Your Own Model

Place your sign images in `data/`, one folder per class (e.g. `data/HELLO/`, `data/THANK_YOU/`).

```bash
python build_model.py   # extract features → saves .npy + labels
python train_model.py   # train KNN → saves hybrid_model.pkl
```

---

## Project Layout

```
VaaniVerse/
├── app.py
├── build_model.py
├── train_model.py
├── hybrid_model.pkl
├── hybrid_labels.txt
├── data/
├── hybrid_data/
└── images/
    └── ISL_CSLRT_Corpus/
        ├── Sentence_Level/
        ├── Word_Level/
        └── Letter_Level/
```

---

## Known Limitations

The sign recognizer needs a clear view of both hands and your face. Poor lighting noticeably drops accuracy. Motion-based features are stubbed out for now — the current model only uses static hand geometry. Accuracy is also tied to how much training data you have per class.

---

## What's Next

Temporal modeling for motion-dependent signs, a built-in dataset recording tool, confidence score display, a transformer-based recognizer to replace KNN, and eventually a web or mobile port.

---

## Tech Stack

Python 3.11 · Tkinter · OpenCV · MediaPipe · NumPy · Scikit-Learn · Google Speech API · googletrans · gTTS · Joblib
