from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})



MODEL_ENDPOINTS = {
    "svm": "http://localhost:5001/predict",  
    "vgg": "http://localhost:5002/predict"   
}

@app.route('/predict', methods=['POST'])
def predict():
    # Get model choice from the request form
    model_choice = request.form.get('model')  
    file = request.files['file']

    if model_choice not in MODEL_ENDPOINTS:
        return jsonify({"error": "Invalid model choice"}), 400

    response = requests.post(
        MODEL_ENDPOINTS[model_choice],
        files={'file': file}
    )

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5000, debug=True) 
