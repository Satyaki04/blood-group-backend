# import os
# import json
# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf

# # File paths (ensure these exist in the backend folder)
# MODEL_PATH = 'fingerprint_blood_group_model.h5'
# LABELS_PATH = 'class_labels.json'

# # Load the model and class label mappings
# model = tf.keras.models.load_model(MODEL_PATH)
# with open(LABELS_PATH, 'r') as f:
#     class_labels = json.load(f)
# # Convert keys to integers if needed
# class_labels = {int(k): v for k, v in class_labels.items()}

# IMG_HEIGHT, IMG_WIDTH = 64, 64

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route('/')
# def index():
#     return 'Fingerprint Blood Group Prediction API is Running!'

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
#     img_file = request.files['image']

#     try:
#         img = Image.open(img_file.stream).convert('RGB')
#         img = img.resize((IMG_WIDTH, IMG_HEIGHT))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#     except Exception as e:
#         return jsonify({'error': f'Invalid image: {str(e)}'}), 400

#     # Prediction
#     try:
#         prediction = model.predict(img_array)[0]
#         pred_idx = int(np.argmax(prediction))
#         pred_label = class_labels[pred_idx]
#         confidence = float(prediction[pred_idx])
#         print('Image shape for prediction:', img_array.shape)
#         print('Raw model prediction:', prediction)

#     except Exception as e:
#         return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

#     return jsonify({
#         'predicted_blood_group': pred_label,
#         'confidence': confidence
#     })

# if __name__ == '__main__':
#     app.run(debug=True)










import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ---- Settings (Update filenames if yours differ) ----
MODEL_PATH = 'fingerprint_blood_group_model.h5'
LABELS_PATH = 'class_labels.json'
IMG_SIZE = (64, 64)  # Your trained model's input size

# ---- Load Model and Labels on Startup ----
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# ---- Flask App Init ----
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/')
def index():
    return 'Fingerprint Blood Group Prediction API is Running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']

    try:
        # PIL for robust image reading
        img = Image.open(img_file.stream).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    try:
        predictions = model.predict(img_array)
        pred_idx = int(np.argmax(predictions[0]))
        pred_label = class_labels[pred_idx]
        confidence = float(np.max(predictions[0]))
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    print({
        'predicted_blood_group': pred_label,
        'confidence': confidence
    })
    return jsonify({
        'predicted_blood_group': pred_label,
        'confidence': confidence
    })



if __name__ == '__main__':
    app.run(debug=True)
