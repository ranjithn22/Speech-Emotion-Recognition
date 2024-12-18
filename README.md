# Speech Emotion Recognition Using Deep Learning

## Overview
This project uses deep learning techniques to recognize emotions from speech audio. The model is trained on the **TESS** and **RAVDESS** datasets and classifies speech into different emotional categories such as angry, happy, sad, fearful, and more. The model achieves **90% accuracy**.

## Features
- **Data Preprocessing**: Audio files are converted into Mel spectrograms for feature extraction.
- **Model**: A Convolutional Neural Network (CNN) is used for emotion classification.
- **Interface**: An easy-to-use interface allows users to test the model with custom audio files.

## Datasets Used
1. **TESS (Toronto Emotional Speech Set)**: A dataset of emotional speech recordings from Canadian English speakers.
2. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**: A dataset containing emotional speech and singing samples.

## Installation

### Requirements
To run this project, you need to have Python 3.x installed along with the following libraries:
- `numpy`
- `pandas`
- `librosa`
- `matplotlib`
- `keras`
- `tensorflow`
- `sklearn`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
