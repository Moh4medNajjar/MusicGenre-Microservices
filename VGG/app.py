from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Load the trained model (replace 'music_genre_model.h5' with the path to your VGG-based model)
model = load_model('music_genre_model.h5')

# Define the genres (ensure this matches the classes used during training)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to preprocess the audio for the model
def preprocess_audio(file_path):
    # Load and preprocess the audio file
    y, sr = librosa.load(file_path, sr=22050)  # Ensure sample rate is the same as during training
    y, _ = librosa.effects.trim(y)  # Trim silence
    
    # Convert audio to Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Pad or truncate to consistent shape
    padded_mel = pad_sequences([mel_spectrogram.T], maxlen=43, padding='post', truncating='post')
    
    return np.expand_dims(padded_mel, axis=-1)  # Shape to (1, 43, 128, 1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = './temp_audio.wav'
    file.save(file_path)

    try:
        # Preprocess audio and predict genre
        audio_features = preprocess_audio(file_path)
        prediction = model.predict(audio_features)
        predicted_genre = GENRES[np.argmax(prediction)]

        result = {
            "predicted_genre": predicted_genre,
        }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        # Clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
