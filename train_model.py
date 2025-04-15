import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models/scalers', exist_ok=True)

def train_crop_model():
    print("Loading crop data...")
    crop_data = pd.read_csv('Crop_recommendation.csv')
    
    # Prepare data
    X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values.astype('float32')
    y = crop_data['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype('int32')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Multi-class Classification Neural Network
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Train
    print("Training crop model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=5),
            ReduceLROnPlateau(factor=0.2, patience=3)
        ],
        verbose=1
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Crop Model Accuracy: {accuracy:.4f}")
    
    # Save
    save_model(model, 'models/crop_model.h5')
    joblib.dump(scaler, 'models/scalers/crop_scaler.pkl')
    joblib.dump(label_encoder, 'models/crop_label_encoder.pkl')

def train_yield_model():
    print("Loading yield data...")
    yield_data = pd.read_csv('Crop_Yield_with_Soil_and_Weather.csv')
    
    # Prepare data
    X = yield_data[['Fertilizer', 'temp', 'N', 'P', 'K']].values.astype('float32')
    y = yield_data['yeild'].values.astype('float32')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Regression Neural Network
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(0.001),
                loss='mean_squared_error',
                metrics=['mae'])
    
    # Train
    print("Training yield model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=5),
            ReduceLROnPlateau(factor=0.2, patience=3)
        ],
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Yield Model - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Save
    save_model(model, 'models/yield_model.h5')
    joblib.dump(scaler, 'models/scalers/yield_scaler.pkl')

if __name__ == '__main__':
    tf.keras.backend.set_floatx('float32')
    train_crop_model()
    train_yield_model()
    print("Training complete. Models saved to /models directory")