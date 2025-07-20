# 🎤 Speech Emotion Recognition (SER) using Amharic Dataset

*Author:* SEKINA HUSSEIN SHERIF
*Date:* July 2025  
*Thesis Base Project* | 

Detecting emotions from human speech using machine learning
This project serves as a key part of my AI thesis 

---


## 📌Project Overview
This notebook classifies human emotions from voice using MFCC features and a MLP model trained on the ASED(Amharic Speech Emotion Detection) dataset.

It builds on an existing Kaggle-baased appproach, improved by fine-tuning features and architecture - resulting in  **26.19% accuracy improvement** (final accuracy **89.15%**)
---


## 📂Dataset

- **Source** - [ASED on Github](https://github.com/Ethio2021/ASED_V1)
- **Access** - Cloned into data/
- **Format** - `.wav` audio files with emotion labels embedded in the filename.
- **Classes Used** -
      - 🎭 Neutral  
      - 😊 Happy  
      - 😢 Sad  
      - 😡 Angry  
      - 😨 Fearful  

  >Note: The raw dataset is not included in this repository due to size and licesnsing. Your `.wav` files to the `/data` folder with the first block of code.

  ---

  ## 🧠 Methodology

- *Audio Feature Extraction*: librosa to compute *40 MFCCs* per sample  
- *Preprocessing*: StandardScaler for normalization, LabelEncoder for labels  
- *Model Architecture*:
  - Built a 1D CNN with:
          4 × Conv1D layers (ReLU activation)
          MaxPooling after each conv layer
          Dropout for regularization
          Flattened and passed through dense layers
          Output layer: Softmax (5 emotion classes)
          Optimizer: Adam
          Loss function: Categorical Cross-Entropy 
          Output: Softmax layer (5 classes)

- *Training*: 50 epochs, batch size 32
- *Evaluation*: Accuracy, precision/recall, and confusion matrix

---

## 📊 Results

| Metric         | Value   |
|----------------|---------|
| Baseline Accuracy | ~62.96%    |
| Final Accuracy    | *~89.15%* |
| Accuracy Gain     | +26.19%     |

### 🔍 Observations

- Highest confusion between fearful and neutral  
- MFCC dimensionality boost and scaling improved generalization  
- Model is lightweight and Colab-friendly

---

## 🧪 How to Run

### 🔧 Setup (Locally or on Colab)
1. Install dependencies:



