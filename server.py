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

# --- FLASK SETUP & PERSISTENCE CONFIG ---
app = Flask(__name__)
CORS(app)

# 🎯 CRITICAL CHANGE: Use the absolute mount path for Railway Volume
PERSISTENT_UPLOAD_PATH = '/data/datasets' 

# Create the directory on the mounted volume (crucial on container startup)
try:
    os.makedirs(PERSISTENT_UPLOAD_PATH, exist_ok=True)
    print(f"✅ Persistent storage setup: Using directory {PERSISTENT_UPLOAD_PATH}")
except Exception as e:
    print(f"❌ ERROR: Could not create persistent directory {PERSISTENT_UPLOAD_PATH}. Check Railway Volume setup. Details: {e}")

# Update the app configuration to use the persistent path
app.config['UPLOAD_FOLDER'] = PERSISTENT_UPLOAD_PATH

# --- MODEL CONFIG ---
MODEL_SAVE_PATH = "cnn_oil_detector.h5"
REG_MODEL_PATH = "dispersion_regressor.joblib"
# *** CRITICAL CHANGE: Now pointing to the weights file ***
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

        # --- FINAL FIX: MANUALLY BUILD ARCHITECTURE AND LOAD WEIGHTS ONLY ---
        print(f"⚠️ Attempting FINAL FIX: Manual EfficientNet architecture build and 'load_weights' from {SPECIES_MODEL_PATH}")
        
        num_classes = len(SELECTED_CLASSES)
        img_height, img_width = SPECIES_IMG_SIZE
        
        # 1. Build the exact architecture used in your Colab script
        base_model = EfficientNetB0(
            input_shape=(img_height, img_width, 3), # Enforcing the correct 3 channels
            include_top=False,
            weights=None # <--- CRITICAL FIX: Set to None to prevent 1-channel initialization error
        )
        
        # Freeze layers as per your training script (optional for loading, but good practice)
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
        
        # 2. Compile the model (needed before loading weights in some TF versions)
        species_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 3. Load ONLY the weights from the .h5 file onto the correct architecture
        # This will load your 3-channel weights successfully now that the architecture is correct.
        species_model.load_weights(SPECIES_MODEL_PATH)
        
        # --- DEBUGGING STEP: Print the final model input/output info ---
        print(f"✅ Fish Species Classifier loaded successfully by restoring weights onto correct architecture.")
        print(f"Model Input Shape forced to: {species_model.input_shape}")
        
    except Exception as e:
        # Capture and print the full traceback for detailed debugging
        print(f"❌ CRITICAL ERROR DURING MODEL LOADING ({SPECIES_MODEL_PATH}):")
        traceback.print_exc() 
        print(f"--------------------------------------------------")
        print(f"❌ Error loading Fish Species Classifier. Details: {e}")


# Call model loading function at startup
load_models()

# --- IMAGE PREPROCESSING (Unchanged, ensures 3-channel input) ---
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
        # CRITICAL: Always convert to "RGB" to ensure 3 channels
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
# --- API ENDPOINTS (File Handling is now persistent) ---
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
    """Handles file uploads and saves them persistently to the Railway Volume."""
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    if file:
        filename = file.filename
        # File is saved to: /data/datasets/<filename> (the persistent path)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(f"💾 File {filename} saved persistently to {app.config['UPLOAD_FOLDER']}")
        return jsonify({'message': f'File {filename} uploaded and saved successfully!'}), 201
    return jsonify({'message': 'An unknown error occurred during upload.'}), 500


@app.route('/api/datasets/list', methods=['GET'])
def list_files():
    """Returns a list of all files in the UPLOAD_FOLDER for persistence."""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        # Filter out non-files (like .DS_Store or directories, although they shouldn't be there)
        files = [f for f in files if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f)) and not f.startswith('.')]
        return jsonify(files)
    except Exception as e:
        return jsonify({'message': f'Error listing files: {str(e)}'}), 500

@app.route('/api/datasets/<filename>', methods=['GET'])
def get_file_content(filename):
    """Serves the content of a specific file from the persistent volume."""
    try:
        # Serves the file from /data/datasets/
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        return jsonify({'message': f'File not found or error: {str(e)}'}), 404


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Endpoint using your specific rule-based analysis logic."""
    data = request.get_json()
    temp_str = data.get('temp')
    acid_str = data.get('acid')
    cond = data.get('condition')
    
    if not temp_str or not acid_str or not cond:
        return jsonify({'message': 'Missing temperature, pH, or condition.'}), 400

    try:
        temp = float(temp_str)
        acid = float(acid_str)
    except ValueError:
        return jsonify({'message': 'Invalid numeric input for temperature or pH.'}), 400

    insight = ""

    # --- RULE-BASED LOGIC ---
    if cond == "Migration Pattern":
        if temp < 10: 
            insight += "Cold temperatures slow migration, altering predator-prey timing.\n"
        elif temp > 28: 
            insight += "High temperatures may cause earlier migration or habitat shifts.\n"
        else: 
            insight += "Temperature supports normal migration patterns.\n"
        if acid < 7.9: 
            insight += "Low pH reduces prey availability for migratory species.\n"
        elif acid > 8.2: 
            insight += "High pH may alter navigational cues.\n"
        else: 
            insight += "pH is suitable for normal migration.\n"
            
    elif cond == "Coral Bleaching":
        if temp > 30: 
            insight += "🚨 CRITICAL: High temperature increases coral bleaching risk.\n"
        else: 
            insight += "Temperature within tolerable range for corals.\n"
        if acid < 8.1: 
            insight += "⚠️ WARNING: Acidification weakens coral skeletons and resilience.\n"
        else: 
            insight += "pH is favorable for coral health.\n"
            
    elif cond == "Species Distribution":
        if temp < 10 or temp > 28: 
            insight += "Temperature shifts may cause species to move to new regions.\n"
        else: 
            insight += "Temperature supports current species distribution.\n"
        if acid < 7.9: 
            insight += "Acidification may disrupt ecosystem balance.\n"
        else: 
            insight += "pH is suitable for species survival.\n"
            
    elif cond == "Healthy Ecosystem":
        if temp > 28: 
            insight += "High temperature can stress species and destabilize ecosystems.\n"
        else: 
            insight += "Temperature is supporting ecosystem stability.\n"
        if acid == -999: # Placeholder for Oil Spill impact 
            insight += "Temperature, pH, and oxygen levels affect oil toxicity, persistence, and recovery.\n"
    
    final_analysis = f"Analysis for {cond}:\n\n{insight.strip()}"

    return jsonify({'analysis': final_analysis})

@app.route('/api/analyze_spill', methods=['POST'])
def analyze_spill():
    if cnn_model is None or reg_pipeline is None:
        return jsonify({"error": "Oil Spill ML Models are not loaded on the server."}), 500

    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({"error": "Missing image_data in request"}), 400

    # 1. Image Processing & CNN Prediction
    img_tensor = preprocess_image_from_base64(data['image_data'])
    if img_tensor is None:
        return jsonify({"error": "Failed to decode or process image data for oil spill detection"}), 400

    pred = cnn_model.predict(img_tensor, verbose=0)
    confidence = float(pred[0][0])
    is_oil = confidence > 0.5
    
    # 2. Environmental Factors 
    wind_speed = float(data.get('wind_speed', 5.0))
    wave_speed = float(data.get('wave_period', 6.0)) 
    latitude = float(data.get('latitude', 10.0))
    
    K_val = 50.0 
    initial_area_km2 = 0.1

    obs = {
        'wind_speed': wind_speed,
        'wave_period': wave_speed,  
        'latitude': latitude,
        'K': K_val,
    }

    forecast = None
    if is_oil:
        forecast = predict_dispersion_daywise(obs, reg_pipeline, REG_FEATURES, initial_area_km2)

    # 3. Interpret Factors for Summary
    wind = "Low wind" if wind_speed <= 6 else ("Moderate wind" if wind_speed <= 12 else "High wind")
    wave = "Slow wave speed" if wave_speed <= 4 else ("Moderate wave speed" if wave_speed <= 8 else "Fast wave speed")
    mixing = "Moderate mixing (Fixed Default)" 

    # 4. Return Combined Result
    return jsonify({
        "oil_detected": is_oil,
        "confidence": round(confidence, 3),
        "environmental_summary": f"Conditions: {wind.lower()}, {wave.lower()}, {mixing.lower()}.",
        "dispersion_forecast": forecast
    })


@app.route('/api/predict_species', methods=['POST'])
def predict_species():
    """
    Handles the fish species prediction request using a separate model and preprocessing chain.
    """
    if species_model is None:
        return jsonify({"error": "Fish Species model failed to load on server start. Check model path and ensure 'model.weights.h5' is present."}), 503

    try:
        data = request.get_json()
        image_data = data.get('image_data')
        
        threshold = float(data.get('threshold', 50.0)) 
        
        if not image_data:
            return jsonify({"error": "No image data provided in request."}), 400

        # Preprocess the image using the classification-specific function
        img_tensor = preprocess_image_for_classification(image_data)
        
        if img_tensor is None:
             return jsonify({"error": "Failed to decode or preprocess image for classification."}), 400

        # Make prediction
        predictions = species_model.predict(img_tensor, verbose=0)
        
        expected_classes = len(SELECTED_CLASSES)
        if predictions.shape[1] != expected_classes:
             print(f"ERROR: Model output shape {predictions.shape} does not match expected classes {expected_classes}.")
             return jsonify({"error": f"Model output mismatch. Expected {expected_classes} classes, got {predictions.shape[1]}"}), 500
        
        confidence = float(np.max(predictions[0]) * 100) 
        pred_index = np.argmax(predictions[0])

        # Apply threshold logic
        if confidence < threshold:
            predicted_class = "Invasive Species"
        else:
            predicted_class = SELECTED_CLASSES[pred_index] 

        # Return the results
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2), 
            "threshold_used": threshold
        })

    except Exception as e:
        traceback.print_exc()
        print(f"--- DETAILED ERROR TRACEBACK ABOVE ---")
        return jsonify({"error": f"Internal server error during species analysis: {e}"}), 500
    
# --- RUN SERVER ---
if __name__ == '__main__':
    print(f"Server running on http://127.0.0.1:5000 (Local Debug Mode)")
    app.run(debug=True, port=5000, use_reloader=False)

# NOTE: For Gunicorn/Railway deployment, the command 'gunicorn server:app' 
# handles the startup, making the 'if __name__ == "__main__":' block for local use only.
