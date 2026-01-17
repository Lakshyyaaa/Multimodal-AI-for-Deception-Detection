# Multimodal-AI-for-Deception-Detection
Multimodal deception detection using facial landmarks and MFCC-based audio features, followed by a custom neural network classifier. The system uses feature extraction without transfer learning.

# Multimodal Deception Detection Using Audio & Visual Features

## Overview

This project implements a **multimodal deception detection system** using **audio and visual cues** extracted from interview videos. Facial landmark features are extracted from video frames, and acoustic features are extracted from audio signals. These features are then normalized and passed to a **custom neural network classifier** trained from scratch.

**Note:** This project uses **feature extraction**, not transfer learning. No pretrained deep learning models are fine-tuned or reused for classification.


## 🧠 Methodology

### 🎥 Visual Feature Extraction

* Facial landmarks are extracted from video frames using **MediaPipe Face Mesh**
* Each frame yields **468 facial landmarks**
* Landmarks are collected across a fixed number of frames per video
* Features are normalized per video sequence

### 🔊 Audio Feature Extraction

* Audio is extracted directly from video files
* **MFCC (Mel-Frequency Cepstral Coefficients)** are computed using Librosa
* Per-coefficient normalization is applied
* MFCC features capture speech characteristics relevant to deception cues

### 🔗 Feature Fusion

* Audio and visual features are aligned at the feature level
* The fused feature representation is passed to a custom neural network

### 🤖 Classification

* A **custom neural network** is trained on the extracted features
* The network is trained **from scratch**
* No pretrained models or transfer learning techniques are used


## 🏗️ Project Structure

```
.
├── Court_Room_Data/
│   ├── Train/
│   │   ├── *.mp4
│
├── features/
│   ├── video/
│   │   └── train/
│   │       └── *.npy
│   ├── audio/
│   │   └── train/
│   │       └── *.npy
│
├── extract_video_features.py
├── extract_audio_features.py
├── train_model.py
├── requirements.txt
└── README.md
```

## 📊 Feature Shapes

### Visual Features

```
(T, 468, 2)
T = number of video frames used per sample
```

### Audio Features

```
(13, T)
T = number of audio frames
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Extract Visual Features

```bash
python extract_video_features.py
```

### Extract Audio Features

```bash
python extract_audio_features.py
```

### Train the Model

```bash
python train_model.py
```

---

## Not Transfer Learning (Important)

This project **does not** use:

* Fine-tuning of pretrained deep networks
* Frozen CNN backbones
* Pretrained audio encoders (e.g., wav2vec, VGGish)
* End-to-end deep learning models

MediaPipe and MFCC extraction are used **only as feature extractors**, and all learning occurs in the custom classifier.

## 📈 Applications

* Deception detection in interviews
* Behavioral analysis
* Multimodal affective computing
* Human-computer interaction research

## 🔮 Future Work

* Temporal modeling using LSTMs or Transformers
* End-to-end multimodal deep learning
* Pretrained audio-visual encoders
* Attention-based feature fusion
* Real-time inference

## 🧑‍💻 Author

**Lakshya Santani**

## 📜 License

This project is intended for academic and research use.

