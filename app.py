from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os

app = Flask(__name__)

def load_models():
    models = {
        'status': False,
        'error': None
    }
    
    try:
        # Load crop models
        models['crop_model'] = load_model('models/crop_model.h5')
        models['crop_scaler'] = joblib.load('models/scalers/crop_scaler.pkl')
        models['crop_encoder'] = joblib.load('models/crop_label_encoder.pkl')
        
        # Load yield models
        models['yield_model'] = load_model('models/yield_model.h5', custom_objects={'mse': MeanSquaredError()})
        models['yield_scaler'] = joblib.load('models/scalers/yield_scaler.pkl')
        
        models['status'] = True
        
    except Exception as e:
        models['error'] = str(e)
        print(f"Error loading models: {models['error']}")
    
    return models

models = load_models()

@app.route('/')
def index():
    return render_template('index.html', 
                         models_loaded=models['status'],
                         error=models.get('error'))

@app.route('/predict', methods=['POST'])
def predict():
    if not models['status']:
        return jsonify({
            'status': 'error',
            'message': models.get('error', 'Models not loaded')
        })
    
    try:
        data = request.get_json()
        
        # Crop recommendation
        crop_input = np.array([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])
        
        crop_input_scaled = models['crop_scaler'].transform(crop_input)
        crop_pred = models['crop_model'].predict(crop_input_scaled)
        recommended_crop = models['crop_encoder'].inverse_transform([np.argmax(crop_pred)])[0]
        
        # Yield prediction
        yield_input = np.array([[
            float(data['fertilizer']),
            float(data['temperature']),
            float(data['N']),
            float(data['P']),
            float(data['K'])
        ]])
        
        yield_input_scaled = models['yield_scaler'].transform(yield_input)
        predicted_yield = float(models['yield_model'].predict(yield_input_scaled)[0][0])
        
        return jsonify({
            'status': 'success',
            'recommended_crop': recommended_crop,
            'predicted_yield': predicted_yield
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)