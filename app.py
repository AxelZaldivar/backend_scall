import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permite todas las solicitudes

# Cargar el modelo
model = keras.models.load_model("model.h5")

@app.route("/predict", methods=["OPTIONS"])
def handle_options():
    return '', 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("inputs", [])
    if not data or not isinstance(data, list):
        return jsonify({"error": "Entrada inválida"}), 400

    inputs = np.array(data)
    prediction = model.predict(inputs).tolist()
    return jsonify({"predictions": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Puerto dinámico para Railway
    app.run(host="0.0.0.0", port=port)
