from flask import Flask, request, jsonify, render_template
from utils import predict_category
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "no selected file"}), 400

    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    label, confidence = predict_category(filepath)
    response = {
        "category": label,
        "confidence": float(confidence)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
