import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import joblib
import base64
import io
import math
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# Import the specific layers and functions needed for model reconstruction
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Sequential 
import traceback 

# --- Suppress oneDNN Warning (as requested) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app)

# ❌ REMOVED: All UPLOAD_FOLDER and volume setup logic 
# The file upload endpoints will be updated in Step 2.

# --- MODEL CONFIG ---
MODEL_SAVE_PATH = "cnn_oil_detector.h5"
REG_MODEL_PATH = "dispersion_regressor.joblib"
SPECIES_MODEL_PATH = 'model.weights.h5' 

OIL_IMG_SIZE = (128, 128)
SPECIES_IMG_SIZE = (224, 224) 
REG_FEATURES = ['wind_speed', 'wave_period', 'group_speed', 'K', 'latitude', 'day']
SELECTED_CLASSES = [
    "Freshwater Eel","GoldFish","Gourami","Ornate sleeper","Perch",
    "Scat fish","Silver Perch","Silver-Body","SnakeHead","Tenpounder","Tilapia"
]

# --- GLOBAL MODELS ---
cnn_model = None
reg_pipeline = None
species_model = None 

def load_models():
    """Attempts to load all necessary ML models."""
    global cnn_model, reg_pipeline, species_model
    
    # 1. Load Oil Spill CNN and Regressor (unchanged)
    try:
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Missing {MODEL_SAVE_PATH}")
        if not os.path.exists(REG_MODEL_PATH):
            raise FileNotFoundError(f"Missing {REG_MODEL_PATH}")

        # Suppress Keras/TF loading output
        cnn_model = tf.keras.models.load_model(MODEL_SAVE_PATH) 
        reg_pipeline = joblib.load(REG_MODEL_PATH)
        print("✅ ML Models (Oil Spill CNN, Regressor) loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Oil Spill ML models. Details: {e}")

    # 2. Load Fish Species Classifier - WEIGHTS ONLY RECOVERY
    try:
        if not os.path.exists(SPECIES_MODEL_PATH):
            raise FileNotFoundError(f"Missing weights file at: {SPECIES_MODEL_PATH}")

        print(f"⚠️ Attempting FINAL FIX: Manual EfficientNet architecture build and 'load_weights' from {SPECIES_MODEL_PATH}")
        
        num_classes = len(SELECTED_CLASSES)
        img_height, img_width = SPECIES_IMG_SIZE
        
        # 1. Build the exact architecture used in your Colab script
        base_model = EfficientNetB0(
            input_shape=(img_height, img_width, 3), 
            include_top=False,
            weights=None 
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Build the sequential model
        species_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        # 2. Compile the model
        species_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 3. Load ONLY the weights
        species_model.load_weights(SPECIES_MODEL_PATH)
        
        print(f"✅ Fish Species Classifier loaded successfully by restoring weights onto correct architecture.")
        print(f"Model Input Shape forced to: {species_model.input_shape}")
        
    except Exception as e:
        traceback.print_exc() 
        print(f"--------------------------------------------------")
        print(f"❌ Error loading Fish Species Classifier. Details: {e}")


# Call model loading function at startup
load_models()

# --- IMAGE PREPROCESSING (Unchanged) ---
def preprocess_image_from_base64(base64_string):
    """Decodes base64 image string to a normalized tensor for the OIL SPILL CNN (128x128)."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_resized = image.resize(OIL_IMG_SIZE)
        img_array = np.array(image_resized) / 255.0 
        img_tensor = tf.expand_dims(img_array, axis=0)
        return img_tensor
    except Exception as e:
        print(f"Oil Spill image decoding error: {e}")
        return None

def preprocess_image_for_classification(base64_data):
    """
    Decodes base64 image string, resizes to 224x224, and applies 
    application-specific normalization for the SPECIES CLASSIFIER.
    """
    try:
        if ',' in base64_data:
            _, base64_str = base64_data.split(',', 1)
        else:
            base64_str = base64_data

        img_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB") 
        image = image.resize(SPECIES_IMG_SIZE)
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = preprocess_input(img_array)
        
        return img_tensor

    except Exception as e:
        print(f"Classification image preprocessing error: {e}")
        return None

# --- DISPERSION FORECAST (Unchanged) ---
def predict_dispersion_daywise(obs, pipeline, features, initial_area_km2, horizon_days=[1, 3, 7]):
    """Calculates the dispersion forecast."""
    if pipeline is None or len(features) == 0:
        return None

    max_h = max(horizon_days)
    rows = []
    
    for t in range(1, max_h + 1):
        row = {}
        for f in features:
            if f in obs and obs[f] is not None:
                row[f] = obs[f]
            elif f == 'day':
                row[f] = t
            elif f == 'group_speed':
                row[f] = 0.0 
            else:
                row[f] = 0.0 
        rows.append(row)

    try:
        Xpred = pd.DataFrame(rows)[features].astype(float)
        ypred = pipeline.predict(Xpred)
        pred_vals = np.maximum(np.array(ypred).reshape(-1,), 0.0)
        cumulative_added = np.cumsum(pred_vals)
        
        out = pd.DataFrame({
            'day': np.arange(1, len(pred_vals) + 1),
            'area_km2_total': (initial_area_km2 + cumulative_added).round(2)
        })
        out = out[out['day'].isin(horizon_days)]
        return out.to_dict('records')
    except Exception as e:
        print(f"Regression error: {e}")
        return None

# =========================================================
# --- API ENDPOINTS (Datasets are now placeheld/disabled) ---
# =========================================================

MOCK_TREND_DATA = {
    'Goa': {
        'Sea Surface Temperature': [28.5, 28.6, 28.7, 28.8, 28.9, 29, 29.1, 29.2, 29.3, 29.4],
        'Ocean Acidification': [8.1, 8.12, 8.11, 8.1, 8.09, 8.08, 8.07, 8.06, 8.05, 8.04],
        'Change in Ocean Currents':[20,22,21,23,22,24,23,25,24,26],
        'Coral Reef Decline': [80, 82, 83, 85, 86, 88, 89, 90, 91, 92],
        'Microplastic Pollution':[50,52,53,55,56,57,58,60,61,62],
        'Ecosystem Degradation':[40,42,43,44,45,46,47,48,49,50]
    },
    'Chennai': {
        'Sea Surface Temperature': [29.3, 29.4, 29.5, 29.6, 29.5, 29.4, 29.3, 29.2, 29.1, 29],
        'Ocean Acidification': [8.0, 8.01, 8.02, 8.03, 8.02, 8.01, 8.0, 7.99, 7.98, 7.97],
        'Change in Ocean Currents':[25,26,27,28,27,26,25,24,23,22],
        'Coral Reef Decline': [75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
        'Microplastic Pollution':[40,41,42,43,44,45,46,47,48,49],
        'Ecosystem Degradation':[35,36,37,38,39,40,41,42,43,44]
    }
}

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Returns all mock trend data for the frontend to manage."""
    return jsonify({
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
        'data': MOCK_TREND_DATA
    })


@app.route('/api/datasets', methods=['POST'])
def upload_file():
    """⚠️ PLACEHOLDER: This endpoint needs to be updated in Step 2 for MongoDB."""
    file = request.files.get('file')
    if file and file.filename:
        # ❌ CRITICAL: File is NOT being saved persistently here!
        print(f"⚠️ Warning: File '{file.filename}' received but NOT saved. Implementing MongoDB next.")
        return jsonify({'message': f'File {file.filename} received but not saved. DB logic required.'}), 201
    return jsonify({'message': 'No file selected for upload.'}), 400


@app.route('/api/datasets/list', methods=['GET'])
def list_files():
    """⚠️ PLACEHOLDER: Returns an empty list until MongoDB is implemented."""
    print("⚠️ Warning: Listing files from MongoDB required.")
    return jsonify([]) # Return an empty list temporarily


@app.route('/api/datasets/<filename>', methods=['GET'])
def get_file_content(filename):
    """⚠️ PLACEHOLDER: Cannot serve content without DB logic."""
    print(f"⚠️ Warning: Attempted to get content for '{filename}'. DB logic required.")
    return jsonify({'message': f'File {filename} content unavailable. Database logic pending.'}), 404


# ... (all other API endpoints: /api/analyze, /api/analyze_spill, /api/predict_species are unchanged)
# ...


@app.route('/api/analyze', methods=['POST'])
# ... (function body unchanged)

# ... (rest of the endpoints unchanged)

@app.route('/api/analyze_spill', methods=['POST'])
# ... (function body unchanged)

@app.route('/api/predict_species', methods=['POST'])
# ... (function body unchanged)


# --- RUN SERVER ---
if __name__ == '__main__':
    print(f"Server running on http://127.0.0.1:5000 (Local Debug Mode)")
    app.run(debug=True, port=5000, use_reloader=False)