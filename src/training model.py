# Import necessary libraries
import os
import numpy as np
import pandas as pd
import librosa  # For audio processing and feature extraction
import matplotlib.pyplot as plt  # For visualization of confusion matrices and training curves
import seaborn as sns  # For better visualization of confusion matrices
from sklearn.metrics import confusion_matrix, classification_report  # For model evaluation metrics
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and test sets
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer  # For encoding emotion labels
from keras.utils import to_categorical  # For converting labels to one-hot encoding
from sklearn.preprocessing import LabelEncoder  # For encoding the emotion labels

# TESS Dataset Parser
def tess_data_parser(filepaths, path):
    """
    Parse the TESS dataset and organize the audio file paths and corresponding emotion labels.

    Parameters:
    filepaths (list): List of folder names representing different emotions.
    path (str): The root directory of the TESS dataset.

    Returns:
    pd.DataFrame: Dataframe with columns: 'emotion', 'classID', 'audio_file'.
    """
    audio_item = []  # List to store paths to audio files
    emotion = []  # List to store emotion labels
    classID = []  # List to store class IDs (numerical representation of emotions)

    # Iterate through each emotion folder and audio files
    for i in filepaths:
        filenames = os.listdir(os.path.join(path, i))
        for f in filenames:
            if i == 'OAF_angry' or i == 'YAF_angry':
                emotion.append('angry')
                classID.append(1)
            elif i == 'OAF_disgust' or i == 'YAF_disgust':
                emotion.append('disgust')
                classID.append(2)
            elif i == 'OAF_Fear' or i == 'YAF_fear':
                emotion.append('fear')
                classID.append(3)
            elif i == 'OAF_happy' or i == 'YAF_happy':
                emotion.append('happy')
                classID.append(4)
            elif i == 'OAF_neutral' or i == 'YAF_neutral':
                emotion.append('neutral')
                classID.append(5)
            elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprise':
                emotion.append('surprise')
                classID.append(6)
            elif i == 'OAF_Sad' or i == 'YAF_sad':
                emotion.append('sad')
                classID.append(7)
            else:
                continue
            audio_item.append(os.path.join(path, i, f))

    # Create a DataFrame to hold the parsed data
    tess_df = pd.DataFrame({'emotion': emotion, 'classID': classID, 'audio_file': audio_item})
    return tess_df

# RAVDESS Dataset Parser
def ravdess_data_parser(path):
    """
    Parse the RAVDESS dataset and organize the audio file paths and corresponding emotion labels.

    Parameters:
    path (str): The root directory of the RAVDESS dataset.

    Returns:
    pd.DataFrame: Dataframe with columns: 'emotion', 'classID', 'audio_file'.
    """
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprise'
    }

    audio_item = []  # List to store paths to audio files
    emotion = []  # List to store emotion labels
    classID = []  # List to store class IDs (numerical representation of emotions)

    # Iterate through all files in the directory and subdirectories
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".wav"):
                emotion_code = filename.split("-")[2]  # Extract emotion code from filename
                emotion_label = emotions[emotion_code]  # Map code to emotion label
                emotion.append(emotion_label)
                classID.append(list(emotions.keys()).index(emotion_code) + 1)  # Class ID starts from 1
                audio_item.append(os.path.join(dirname, filename))  # Add file path

    # Create a DataFrame to hold the parsed data
    ravdess_df = pd.DataFrame({'emotion': emotion, 'classID': classID, 'audio_file': audio_item})
    return ravdess_df

# Function to extract mel spectrograms from audio files
def extract_mel_spectrogram(file, sampling_rate=22050, target_length=128):
    """
    Extract mel spectrogram from an audio file and return it.

    Parameters:
    file (str): Path to the audio file.
    sampling_rate (int): Sampling rate to load the audio.
    target_length (int): Target length of the spectrogram (padded or truncated).

    Returns:
    np.ndarray: The mel spectrogram of the audio.
    """
    # Load the audio file
    y, sr = librosa.load(file, sr=sampling_rate)
    # Compute the mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to dB scale for better visualization

    # Pad or truncate the spectrogram to match target length
    if mel_db.shape[1] < target_length:
        pad_width = target_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_length]  # Trim to target length

    return mel_db

# Load and combine TESS and RAVDESS datasets
tess_data_path = "TESS dataset path"
tess_audio = os.listdir(tess_data_path)
tess_audio.sort()  # Sort the folder names alphabetically
tess_df = tess_data_parser(tess_audio, tess_data_path)

ravdess_path = "ravdess dataset path"
ravdess_df = ravdess_data_parser(ravdess_path)

# Combine both datasets into one DataFrame
combined_df = pd.concat([tess_df, ravdess_df], ignore_index=True)

# Prepare features and labels
mel_data = []  # List to store mel spectrograms
targets = []  # List to store corresponding labels

# Extract features (mel spectrograms) for each audio file
for index, row in combined_df.iterrows():
    mel_spectrogram = extract_mel_spectrogram(row['audio_file'])
    mel_data.append(mel_spectrogram)  # Append extracted mel spectrogram
    targets.append(row['emotion'])  # Append corresponding emotion label

# Convert data to numpy arrays
X = np.array(mel_data)
X = np.expand_dims(X, axis=-1)  # Add channel dimension to match model input
y = np.array(targets)

# One-hot encode labels using LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)  # Convert emotion labels to numeric values
y = to_categorical(y)  # One-hot encoding

# Get number of unique classes (emotions)
num_classes = y.shape[1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the model architecture
def compile_model(input_shape, num_classes):
    """
    Compile and return a CNN model for speech emotion recognition.

    Parameters:
    input_shape (tuple): Shape of the input data.
    num_classes (int): Number of output classes.

    Returns:
    model: A compiled CNN model.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Softmax activation for multi-class classification

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
input_shape = (128, 128, 1)
model = compile_model(input_shape, num_classes)
model.summary()

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)

# Save the model
model.save('speech_emotion_recognition_model.h5')

# Plotting accuracy and loss
plt.figure(figsize=(12, 4))

# Plotting Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
