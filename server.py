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
import traceback 
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# --- MongoDB Imports and Environment Setup ---
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId 

# Load environment variables from .env file
load_dotenv()

# Import the specific layers and functions needed for model reconstruction
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Sequential 

# --- Suppress oneDNN Warning (as requested) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- FLASK SETUP & DB CONFIG ---
app = Flask(__name__)
CORS(app)

# 🚀 MONGO DB CONFIGURATION
MONGO_URI = os.getenv("MONGO_URI") 
DB_NAME = "aquaris_datasets"
COLLECTION_NAME = "datasets"
MAX_CONTENT_SIZE_MB = 10 

# --- DB CONNECTION ---
db_client = None
db = None
try:
    if MONGO_URI:
        db_client = MongoClient(MONGO_URI)
        db = db_client[DB_NAME]
        db.command('ping') 
        print(f"✅ MongoDB connected successfully to database: {DB_NAME}")
    else:
        print("❌ CRITICAL: MONGO_URI not found in environment variables. Persistence disabled.")
except Exception as e:
    print(f"❌ CRITICAL: Failed to connect to MongoDB. Details: {e}")
    db_client = None
    db = None

# --- MODEL CONFIG (Unchanged) ---
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

# --- GLOBAL MODELS (Unchanged) ---
cnn_model = None
reg_pipeline = None
species_model = None 

def load_models():
    """Attempts to load all necessary ML models."""
    global cnn_model, reg_pipeline, species_model
    
    # 1. Load Oil Spill CNN and Regressor
    try:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"File not found: {MODEL_SAVE_PATH}")
        if not os.path.exists(REG_MODEL_PATH):
            print(f"File not found: {REG_MODEL_PATH}")

        # Placeholder/Mocked loading to prevent crash if files are missing in env
        # cnn_model = tf.keras.models.load_model(MODEL_SAVE_PATH) 
        # reg_pipeline = joblib.load(REG_MODEL_PATH)
        # print("✅ ML Models (Oil Spill CNN, Regressor) loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Oil Spill ML models (placeholder logic running). Details: {e}")
    
    # 2. Load Fish Species Classifier - WEIGHTS ONLY RECOVERY
    try:
        if not os.path.exists(SPECIES_MODEL_PATH):
            print(f"Missing weights file at: {SPECIES_MODEL_PATH}")
            return # Skip species model loading

        num_classes = len(SELECTED_CLASSES)
        img_height, img_width = SPECIES_IMG_SIZE
        
        # 1. Build the exact architecture
        base_model = EfficientNetB0(
            input_shape=(img_height, img_width, 3), 
            include_top=False,
            weights=None 
        )
        
        # Freeze layers
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
        
        # 2. Compile and load ONLY the weights
        species_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        species_model.load_weights(SPECIES_MODEL_PATH)
        
        print(f"✅ Fish Species Classifier loaded successfully.")
        
    except Exception as e:
        traceback.print_exc() 
        print(f"❌ Error loading Fish Species Classifier. Details: {e}")


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
        # Mock prediction if reg_pipeline is None
        if pipeline is None:
            ypred = np.random.rand(len(Xpred)) * 0.5 
        else:
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
# --- HELPER FUNCTIONS FOR MONGODB ---
# =========================================================

def read_file_content_for_db(file, max_size_mb):
    """Reads file content as string, checking size limits."""
    file.seek(0)
    
    file_size_bytes = file.content_length # Use Flask's content_length 
    
    if file_size_bytes > max_size_mb * 1024 * 1024:
        print(f"File {file.filename} skipped content storage. Size: {file_size_bytes/1024/1024:.2f}MB > {max_size_mb}MB limit.")
        file.seek(0) # Reset pointer just in case
        return None 

    if file.filename.lower().endswith('.zip'):
        print(f"File {file.filename} skipped content storage (ZIP detected).")
        file.seek(0)
        return None 
    
    try:
        content = file.read().decode('utf-8')
        file.seek(0)
        return content
    except Exception as e:
        print(f"Error reading file content for {file.filename}: {e}")
        file.seek(0)
        return None

# =========================================================
# --- API ENDPOINTS (Datasets are now MongoDB-Persistent) ---
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
    """Handles file uploads and saves METADATA and CONTENT to MongoDB."""
    if db is None:
        return jsonify({'message': 'Database not connected. Cannot upload.'}), 503

    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
        
    if file:
        filename = file.filename
        
        # 1. Read the content (checks size limit and zips)
        file_content = read_file_content_for_db(file, MAX_CONTENT_SIZE_MB) 

        # 2. Save document to MongoDB
        try:
            file_length = file.content_length 

            dataset_doc = {
                "filename": filename,
                "content_type": file.mimetype,
                "size_bytes": file_length,
                "upload_time": pd.Timestamp.now().isoformat(),
                "content": file_content 
            }
            
            # Upsert behavior: Delete existing file with same name, then insert new one
            db[COLLECTION_NAME].delete_one({"filename": filename})
            result = db[COLLECTION_NAME].insert_one(dataset_doc)
            
            status_msg = "successfully!" if file_content is not None else f"successfully, but **content not saved** (too large or ZIP)."
            print(f"💾 File {filename} uploaded to MongoDB with ID: {result.inserted_id}. Status: {status_msg}")
            
            return jsonify({'message': f'File {filename} uploaded and saved to DB {status_msg}'}), 201

        except Exception as e:
            traceback.print_exc()
            return jsonify({'message': f'Error saving to MongoDB: {str(e)}'}), 500

    return jsonify({'message': 'An unknown error occurred during upload.'}), 500


@app.route('/api/datasets/list', methods=['GET'])
def list_files():
    """Returns a list of all file NAMES from MongoDB."""
    if db is None:
        return jsonify([])
        
    try:
        # Fetch only the filename field
        files_cursor = db[COLLECTION_NAME].find({}, {"filename": 1, "_id": 0})
        files = [doc['filename'] for doc in files_cursor]
        return jsonify(files)
    except Exception as e:
        print(f"Error listing files from MongoDB: {e}")
        return jsonify([])

@app.route('/api/datasets/<filename>', methods=['GET'])
def get_file_content(filename):
    """Serves the file content (the 'content' field) from MongoDB."""
    if db is None:
        return jsonify({'message': 'Database not connected. Cannot retrieve content.'}), 503
        
    try:
        # 1. Find the document by filename
        doc = db[COLLECTION_NAME].find_one({"filename": filename})
        if not doc:
            return jsonify({'message': f'File {filename} not found in database.'}), 404

        # 2. Retrieve the content
        content = doc.get("content")
        
        if content is None:
             return jsonify({'message': f'Content for {filename} is not stored in the database. (File too large or ZIP)'}), 404

        # 3. Return the content as text
        # 🐛 FIX: Added missing closing parenthesis here!
        return Response(
            response=content,
            status=200,
            mimetype=doc.get("content_type", 'text/plain')
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error retrieving file content from MongoDB: {str(e)}'}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyzes uploaded CSV or JSON data (mock implementation)."""
    data = request.get_json()
    filename = data.get('filename')
    
    # 1. Retrieve the content from MongoDB
    if db is None:
        return jsonify({'message': 'Database not connected. Cannot analyze.'}), 503

    try:
        doc = db[COLLECTION_NAME].find_one({"filename": filename})
        if not doc or doc.get('content') is None:
            return jsonify({'message': f'File {filename} not found or content not stored.'}), 404
        
        content = doc['content']
        mimetype = doc.get('content_type', 'text/csv')

        # 2. Read the content into a DataFrame
        if 'csv' in mimetype:
            df = pd.read_csv(io.StringIO(content))
        elif 'json' in mimetype:
            df = pd.read_json(io.StringIO(content))
        else:
            return jsonify({'message': 'Unsupported file type for analysis.'}), 400

        # 3. Perform Mock Analysis
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'first_5_rows': df.head().to_dict(orient='records')
        }
        
        return jsonify({
            'message': f'Analysis for {filename} complete.',
            'summary': summary
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error during data analysis: {str(e)}'}), 500


@app.route('/api/analyze_spill', methods=['POST'])
def analyze_spill():
    """Predicts oil spill status based on an image and environmental data."""
    if cnn_model is None or reg_pipeline is None:
        return jsonify({'message': 'ML models are not loaded. Cannot analyze spill.'}), 503
        
    data = request.get_json()
    base64_image = data.get('image')
    obs = data.get('observation', {})
    initial_area_km2 = float(obs.get('initial_area', 0.1)) # Default 0.1 km2

    # 1. Image Classification (Oil/No Oil)
    img_tensor = preprocess_image_from_base64(base64_image)
    if img_tensor is None:
        return jsonify({'message': 'Invalid image data.'}), 400
        
    try:
        prediction = cnn_model.predict(img_tensor)
        spill_detected = bool(np.round(prediction[0][0]))
        confidence = float(prediction[0][0])
        status = "Oil Spill Detected" if spill_detected else "No Oil Spill Detected"
        
        # 2. Dispersion Prediction (Only if spill is detected)
        dispersion_forecast = None
        if spill_detected:
            dispersion_forecast = predict_dispersion_daywise(
                obs, reg_pipeline, REG_FEATURES, initial_area_km2
            )
            
        return jsonify({
            'status': status,
            'confidence': confidence,
            'spill_detected': spill_detected,
            'forecast': dispersion_forecast
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error during spill analysis: {str(e)}'}), 500

@app.route('/api/predict_species', methods=['POST'])
def predict_species():
    """Classifies fish species based on an image."""
    if species_model is None:
        return jsonify({'message': 'Fish species model is not loaded.'}), 503
        
    data = request.get_json()
    base64_image = data.get('image')

    # 1. Preprocess Image
    img_tensor = preprocess_image_for_classification(base64_image)
    if img_tensor is None:
        return jsonify({'message': 'Invalid image data or preprocessing failed.'}), 400
        
    try:
        # 2. Predict
        predictions = species_model.predict(img_tensor)[0]
        
        # 3. Format results
        results = []
        top_k_indices = np.argsort(predictions)[::-1][:3] # Top 3 predictions
        
        for i in top_k_indices:
            results.append({
                'species': SELECTED_CLASSES[i],
                'confidence': float(predictions[i])
            })
            
        # Determine the top result
        top_result = results[0]
        
        return jsonify({
            'message': 'Species classification complete.',
            'top_species': top_result['species'],
            'top_confidence': top_result['confidence'],
            'all_results': results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Error during species prediction: {str(e)}'}), 500


# --- RUN SERVER ---
# Line 265 (now fixed)
if __name__ == '__main__':
    print(f"Server running on http://127.0.0.1:5000 (Local Debug Mode)")
    app.run(debug=True, port=5000, use_reloader=False)