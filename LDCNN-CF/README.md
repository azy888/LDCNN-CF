# LDCNN-CF: Lightweight Dual-Branch CNN for Acoustic Comfort Prediction

This repository contains the official implementation of the LDCNN-CF model for predicting acoustic comfort of electric toothbrush sounds.

## Features

- Lightweight dual-branch CNN architecture (0.42M parameters)
- Cross-attention fusion for Mel-spectrogram and psychoacoustic features
- 8-fold leave-one-brand-out cross-validation
- Comprehensive evaluation metrics (MAE, RMSE, MAPE, R²)
- Statistical significance testing (paired t-test, bootstrap CI)
- Attention visualization

## Requirements

```bash
pip install -r requirements.txt