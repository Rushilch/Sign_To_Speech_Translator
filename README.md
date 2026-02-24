# **VaaniVerse — Multilingual Voice ↔ Sign Language Interpreter**

**VaaniVerse** is a real-time **multilingual communication platform** that bridges the gap between **spoken languages and Indian Sign Language (ISL)**.
Built using **Computer Vision, Machine Learning, Speech Recognition, and Translation**, the system enables seamless interaction between hearing-impaired and hearing individuals.

The application is implemented as a **desktop GUI using Tkinter**, with real-time camera input, speech input, and intelligent sign playback.

---

## 🔑 Key Features

### 🎤 Speak → Sign

* Speech input in **multiple Indian languages**
* Automatic translation to **English**
* Hierarchical sign search:

  * **Sentence-level signs**
  * **Word-level signs**
  * **Letter-level fallback**
* Smooth sign playback using images and GIFs

### ✋ Sign → Speech

* Real-time hand & face tracking using **MediaPipe**
* Hybrid feature extraction (hand + face spatial relationship)
* **KNN-based sign recognition**
* Automatic sentence construction
* Translation to selected language
* **Google Text-to-Speech output**

### 🌐 Multilingual Support

Supported languages:

* English
* Hindi
* Telugu
* Tamil
* Bengali
* Marathi
* Gujarati
* Kannada

---

## 🧠 Intelligent Components

### 🔍 Computer Vision

* Hand landmark detection (21 points per hand)
* Face landmark detection (nose reference)
* Feature normalization and spatial encoding

### 🧠 Machine Learning

* Hybrid feature vector (static + motion placeholder)
* **K-Nearest Neighbors (KNN)** classifier
* Distance-weighted prediction
* Stable prediction window for accuracy

### 🗣 Speech & Translation

* Google Speech Recognition
* Google Translate (`googletrans`)
* Google Text-to-Speech (`gTTS`)

---

## 🖥 Interface & UX

* Clean, responsive Tkinter UI
* Two-tab workflow:

  * **Speak to Sign**
  * **Sign to Speech**
* Live camera feed with overlays
* Real-time sentence building
* Translation & phonetic display

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/VaaniVerse.git
cd VaaniVerse
```

### 2️⃣ Create Virtual Environment (Python 3.11 Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib pillow tqdm
pip install SpeechRecognition googletrans==4.0.0-rc1 gtts
```

> ⚠️ Python **3.11** is strongly recommended for MediaPipe stability.

---

## ▶️ Run the Application

```bash
python app.py
```

The application launches a GUI with **Speak → Sign** and **Sign → Speech** modes.

---

## 🧪 Model Training & Dataset Preparation

### 📁 Dataset Structure

```
data/
├── HELLO/
│   ├── img1.jpg
│   ├── img2.jpg
├── THANK_YOU/
├── HOW_ARE_YOU/
```

Each folder name represents **one sign class**.

---

### 🧩 Step 1: Feature Extraction

Run:

```bash
python build_model.py
```

This script:

* Detects hand & face landmarks
* Normalizes hand geometry
* Computes face-relative distances
* Creates a **hybrid feature vector**
* Saves features as `.npy`
* Generates `hybrid_labels.txt`

---

### 🧠 Step 2: Model Training

Run:

```bash
python train_model.py
```

* Trains a **KNN classifier**
* Uses distance-based weighting
* Saves trained model as `hybrid_model.pkl`

---

## 🗂 Project Structure

```
VaaniVerse/
├── app.py                     # Main application
├── build_model.py              # Feature extraction
├── train_model.py              # Model training
├── hybrid_model.pkl            # Trained model
├── hybrid_labels.txt           # Label mapping
├── data/                       # Raw dataset
├── hybrid_data/                # Extracted features
└── images/
    └── ISL_CSLRT_Corpus/
        ├── Sentence_Level/
        ├── Word_Level/
        └── Letter_Level/
```

---

## 🔊 Text-to-Speech Behavior

| Mode             | Output                 |
| ---------------- | ---------------------- |
| Speak (Eng)      | English sentence       |
| Speak (Phonetic) | Native language speech |

Uses **Google TTS** for accurate pronunciation of regional languages.

---

## ⚠️ Notes & Limitations

* Requires **visible hands and face**
* Accuracy depends on dataset quality
* Motion features are placeholders (extensible)
* Requires stable lighting for best results

---

## 🧪 Tech Stack

| Layer       | Technologies          |
| ----------- | --------------------- |
| Language    | Python 3.11           |
| UI          | Tkinter               |
| Vision      | OpenCV, MediaPipe     |
| ML          | NumPy, Scikit-Learn   |
| Speech      | Google Speech API     |
| Translation | googletrans           |
| TTS         | Google Text-to-Speech |
| Storage     | Joblib                |

---

## 🔮 Planned Enhancements

* Temporal motion modeling
* Confidence visualization
* Dataset recording tool
* Transformer-based recognition
* Mobile & web deployment

---