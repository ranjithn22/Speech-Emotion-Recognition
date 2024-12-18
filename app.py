import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa.display

# Define all possible emotions from both datasets
emotions = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load LabelEncoder with combined emotions
lb = LabelEncoder()
lb.fit(emotions)

# Load the trained model
model_path = 'speech_emotion_recognition_model.h5'
model = load_model(model_path)

def predict_emotion(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure the shape matches the model's expected input shape (128, 128, 1)
    print("Original shape:", mel_spectrogram.shape)

    if mel_spectrogram.shape[1] < 128:
        pad_width = 128 - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    mel_spectrogram = mel_spectrogram[:, :128]  # Trim to 128 frames if longer
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension

    print("Resized shape:", mel_spectrogram.shape)

    prediction = model.predict(mel_spectrogram)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = lb.inverse_transform(predicted_class)

    return predicted_emotion[0], prediction

def plot_prediction(prediction, emotions):
    fig, ax = plt.subplots()
    ax.bar(emotions, prediction[0])
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Prediction')
    return fig

def run():
    st.title("Speech Emotion Recognition")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio("temp.wav")

        if st.button("Predict Emotion"):
            predicted_emotion, prediction = predict_emotion("temp.wav")
            st.write(f"Predicted Emotion: {predicted_emotion}")

            fig = plot_prediction(prediction, emotions)
            st.pyplot(fig)

if __name__ == "__main__":
    run()
