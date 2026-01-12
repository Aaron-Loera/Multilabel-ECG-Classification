# Multilabel ECG Classification

This repository contains a comprehensive workflow for multilabel classification of ECG signals using the PTB-XL database. The project consists of three main Jupyter notebooks that handle data preprocessing, model development, and feature space analysis.

## Project Overview

This project implements a deep learning approach for multilabel ECG classification using 12-lead ECG signals. The workflow progresses through data preprocessing, custom CNN development, and feature space exploration to provide insights into cardiac abnormality classification.

## Workflow Description

### 1. Data Preprocessing (`Preprocessing.ipynb`)

The first notebook handles the complete data preprocessing pipeline:

- **Database Import**: Loads the original PTB-XL database from PhysioNet
- **Superclass Mapping**: Maps individual SCP codes to their corresponding diagnostic superclasses (CD, HYP, MI, NORM, STTC)
- **Data Filtering**: Removes records without corresponding superclass labels to ensure clean multilabel classification
- **Signal Extraction**: Extracts all 12-lead ECG signals sampled at 100Hz (1000 time points per lead)
- **Data Storage**: Saves processed records and signals in compressed NumPy format (`.npz`) for efficient loading

**Output**: `processed_datasets/processed_dataset_12_lead_lr.npz` containing cleaned metadata and 12-lead ECG signals

### 2. Model Development (`Cross_Lead_CNN.ipynb`)

The second notebook focuses on developing and training a custom multilabel CNN architecture:

- **Custom Architecture**: Implements a cross-lead CNN designed specifically for multilabel ECG classification
- **Iterative Training**: Multiple training sessions with different hyperparameters stored in `training_sessions/` directory
- **Performance Monitoring**: Comprehensive evaluation including:
  - Training history plots (loss and accuracy curves)
  - Confusion matrices for each class
  - Detailed classification reports with precision, recall, and F1-scores
- **Model Persistence**: Saves trained model weights for each epoch/iteration for reproducibility

**Output**: Trained CNN models with complete performance analytics across multiple training sessions

### 3. Feature Space Analysis (`Feature_Space.ipynb`)

The third notebook explores the learned representations from the pretrained CNN:

- **Embedding Extraction**: Generates feature embeddings from the trained multilabel CNN model
- **Class-based Clustering**: Uses K-Means clustering to visualize feature space organization with respect to diagnostic classes
- **Metadata Correlation**: Analyzes feature space with respect to patient metadata:
  - Age distribution patterns
  - Sex-based clustering
  - Height and weight correlations
- **Linear Probing**: Implements linear classifiers on frozen features to assess:
  - Latent signal encoding quality
  - Information content of learned representations
  - Metadata information preservation during training

**Output**: Comprehensive feature space visualizations and linear probing results stored in `probe_training_sessions/`

## Repository Structure

```
Multilabel-ECG-Classification/
│
├── 📓 Cross_Lead_CNN.ipynb          # CNN model development and training
├── 📓 Feature_Space.ipynb           # Feature space analysis and exploration  
├── 📓 Preprocessing.ipynb           # Data preprocessing and preparation
│
├── 📂 processed_datasets/           # Processed and compressed datasets
│
├── 📂 Scripts/                      # Python modules and utilities
│   ├── cnn.py                      # CNN model architecture definitions
│   ├── pbtxl.py                    # PTB-XL database utilities
│
├── 📂 training_sessions/            # CNN training results and artifacts
│   ├── session_1/                  # Training iteration 1
│   ├── session_2/                  # Training iteration 2
│   ├── session_3/                  
│   ├── session_4/                  
│   └── session_n/                  
│       ├── classification_report.txt     # Performance metrics
│       ├── training_log.txt             # Training progress logs
│       ├── notes.txt                    # Session notes and observations
│       ├── confusion_matrices/          # Confusion matrix visualizations
│       └── weights/                     # Saved model weights per epoch
│           ├── CustomCNN_12Lead_1.weights.h5
│           ├── CustomCNN_12Lead_2.weights.h5
│           └── ...
│
└── 📂 probe_training_sessions/            # Linear probing analysis results
    └── session_1/                        # Feature space probing session 1
        ├── age/                    
        ├── height/                 
        ├── sex/                    
        └── weight/                 
            ├── classification_report.txt  # Probing performance metrics
            ├── training_log.txt           # Probing training logs
            └── weights/                   # Saved probe classifier weights
```

## Key Features

- **Multilabel Classification**: Handles multiple simultaneous cardiac abnormalities
- **12-Lead ECG Processing**: Utilizes all standard ECG leads for comprehensive analysis  
- **Comprehensive Evaluation**: Multiple performance metrics and visualizations
- **Feature Space Analysis**: Deep dive into learned representations
- **Reproducible Experiments**: Organized session management with detailed logging
- **Linear Probing**: Systematic evaluation of feature quality and metadata encoding

## Dependencies

- `wfdb`: For ECG signal processing and PTB-XL database interaction
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations and array operations
- `matplotlib`: Plotting and visualization
- `tensorflow/keras`: Deep learning framework for CNN implementation
- `scikit-learn`: Machine learning utilities and evaluation metrics

## Usage

1. **Data Preparation**: Run `Preprocessing.ipynb` to process the PTB-XL database
2. **Model Training**: Execute `Cross_Lead_CNN.ipynb` to train the multilabel CNN
3. **Feature Analysis**: Use `Feature_Space.ipynb` to explore learned representations

Each notebook contains detailed documentation and can be run sequentially for the complete workflow.

## Database Source

This project uses the PTB-XL database:
- **Citation**: Physikalisch-Technische Bundesanstalt (PTB)
- **Link**: [PTB-XL Database v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- **Description**: Large 12-lead ECG database with diagnostic annotations

## Results

The project generates comprehensive results including:
- Trained multilabel CNN models with performance metrics
- Feature space visualizations and clustering analysis
- Linear probing results demonstrating representation quality
- Detailed training logs and confusion matrices for model interpretation