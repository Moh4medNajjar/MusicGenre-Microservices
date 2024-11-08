# SVM


from flask import Flask, request, jsonify
import librosa
import numpy as np
import os
import joblib  

app = Flask(__name__)

model = joblib.load('model.pkl')

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
        y, sr = librosa.load(file_path, sr=None)

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        spectral_centroid_avg = np.mean(spectral_centroids)
        spectral_rolloff_avg = np.mean(spectral_rolloff)

        features = np.array([spectral_centroid_avg, spectral_rolloff_avg]).reshape(1, -1)

        predicted_genre = model.predict(features)[0]

        result = {
            "predicted_genre": predicted_genre,
        }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=5001)
