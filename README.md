# Heart Murmur Detection with Deep Learning

## Overview
This repository contains the code and experiments for a **heart murmur detection system** using deep learning on phonocardiogram (PCG) recordings.  
The goal is to classify each heart sound recording into one of three categories:
- **Present** – murmur is detected
- **Absent** – no murmur detected
- **Unknown** – uncertain or poor-quality recording

The work builds on the **CirCor DigiScope Phonocardiogram Dataset** and explores multiple neural network architectures, feature extraction techniques, and evaluation settings.

## Dataset
We used the **CirCor DigiScope Phonocardiogram Dataset (PhysioNet v1.0.3)**, which contains heart sound recordings from multiple auscultation locations, collected in clinical settings.  
For most experiments, only the **publicly available training portion** of the dataset was used.  
Multi-location recordings were segmented, preprocessed, and converted into spectrograms before training.

## Models
We evaluated several architectures:
- **CNN-based classifier**
- **ResNet-based models** (ResNet18 and ResNet34, adapted for 1-channel or 3-channel spectrograms)
- **Attention-based CNN classifier**
- **Bi-directional 2-layer LSTM** (baseline sequence model)

## Feature Extraction
- **Mel spectrograms** were extracted from PCG recordings.
- Both **single-window** and **multi-window** spectrogram approaches were tested.
- Normalization and augmentation were applied to improve generalization.

## Experimental Settings
We explored:
- **Patient-dependent** and **patient-independent** evaluation scenarios
- **Class imbalance handling** using weighted cross-entropy and focal loss
- **Adam optimizer** with learning rate scheduling
- Batch sizes between **32** and **64**, trained for **5–100 epochs** depending on the model

## Results
- **Attention-based CNN** with focal loss achieved the highest performance in the patient-independent setting.
- **ResNet34** performed well in the patient-dependent setting.
- LSTM underperformed compared to CNN-based models.

### 📦 Libraries Used

The following Python libraries were used in this project:

- `torch` – for deep learning model training and inference  
- `torchvision` – for ResNet-based architectures  
- `numpy`, `pandas` – for data manipulation and loading  
- `matplotlib`, `seaborn` – for visualizing training progress, evaluation metrics, and spectrograms  
- `scikit-learn` – for computing precision, recall, F1-score, and confusion matrix  
- `librosa` – for audio preprocessing and **Mel spectrogram extraction**  
- `tqdm` – for tracking training progress  
- `os`, `glob`, `argparse` – for file system operations and CLI support  
- `warnings` – for suppressing unnecessary output

## How to Use
You can run the **entire pipeline** with:
```bash
main.py

