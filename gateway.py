from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for the specific frontend origin
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

# Mapping models to their endpoints
MODEL_ENDPOINTS = {
    "svm": "http://localhost:5001/predict",
    "vgg": "http://localhost:5002/predict"
}

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the model choice and file from the request
    model_choice = request.form.get('model')
    file = request.files.get('file')

    if not model_choice or model_choice not in MODEL_ENDPOINTS:
        return jsonify({"error": "Invalid or missing model choice"}), 400

    if not file:
        return jsonify({"error": "File not provided"}), 400

    try:
        # Forward the file to the appropriate model endpoint
        response = requests.post(
            MODEL_ENDPOINTS[model_choice],
            files={'file': file}
        )
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        return jsonify({"error": f"Error communicating with model: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
