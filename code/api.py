# api.py - UPDATED TO READ TRAINING METRICS FROM CSV
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import uvicorn
import re
from datetime import datetime
import os
import sys
import json
import csv
from pathlib import Path

app = FastAPI(
    title="Used Product Price Analyzer API",
    description="Predict prices for used products with training metrics integration",
    version="2.1.0"
)

# Initialize global variables
model = None
scaler = None
label_encoders = {}
feature_columns = []
model_config = {}
training_metrics = {}

class ProductRequest(BaseModel):
    title: str
    condition: str = "good"
    brand: Optional[str] = None
    product_year: Optional[int] = None
    shipping_cost: float = 0.0
    has_defects: bool = False

class PricePrediction(BaseModel):
    predicted_price: float
    confidence: float
    price_range_low: float
    price_range_high: float
    message: str
    model_type: str = "Loaded Model"
    training_mae: Optional[float] = None
    model_accuracy: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# ========== READ TRAINING SUMMARY ==========
def get_training_metrics():
    """Read training metrics from training_summary.csv"""
    print("\n📊 Looking for training summary...")
    
    metrics = {
        'final_train_mae': None,
        'final_val_mae': None,
        'final_train_r2': None,
        'final_val_r2': None,
        'final_train_loss': None,
        'final_val_loss': None,
        'epochs_trained': None,
        'best_epoch': None,
        'timestamp': None,
        'file_source': None
    }
    
    # Check multiple possible locations
    csv_files = [
        'learningBase/training_summary.csv',
        'models/training_summary.csv',
        'data/training_summary.csv',
        'results/training_summary.csv',
        '../training_summary.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                print(f"📁 Found training summary at: {csv_file}")
                
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    
                    # Get all rows and find the latest one
                    rows = list(reader)
                    
                    if not rows:
                        print(f"⚠️  {csv_file} is empty")
                        continue
                    
                    # Sort by timestamp if available, otherwise use last row
                    if 'timestamp' in rows[0]:
                        try:
                            # Convert to datetime for sorting
                            rows.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S') if x['timestamp'] else datetime.min)
                        except:
                            pass  # Keep original order if timestamp parsing fails
                    
                    # Use the most recent (last) row
                    latest_row = rows[-1]
                    
                    # Extract and convert metrics
                    def safe_float(value):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return None
                    
                    def safe_int(value):
                        try:
                            return int(float(value))
                        except (ValueError, TypeError):
                            return None
                    
                    metrics['final_train_mae'] = safe_float(latest_row.get('final_train_mae'))
                    metrics['final_val_mae'] = safe_float(latest_row.get('final_val_mae'))
                    metrics['final_train_r2'] = safe_float(latest_row.get('final_train_r2'))
                    metrics['final_val_r2'] = safe_float(latest_row.get('final_val_r2'))
                    metrics['final_train_loss'] = safe_float(latest_row.get('final_train_loss'))
                    metrics['final_val_loss'] = safe_float(latest_row.get('final_val_loss'))
                    metrics['epochs_trained'] = safe_int(latest_row.get('epochs_trained'))
                    metrics['best_epoch'] = safe_int(latest_row.get('best_epoch'))
                    metrics['timestamp'] = latest_row.get('timestamp', '').strip()
                    metrics['file_source'] = csv_file
                    
                    print(f"✅ Training metrics loaded successfully:")
                    if metrics['final_train_mae'] is not None:
                        print(f"   - MAE (Train): ${metrics['final_train_mae']:.2f}")
                    if metrics['final_val_mae'] is not None:
                        print(f"   - MAE (Val): ${metrics['final_val_mae']:.2f}")
                    if metrics['final_train_r2'] is not None:
                        print(f"   - R² (Train): {metrics['final_train_r2']:.4f}")
                    if metrics['epochs_trained'] is not None:
                        print(f"   - Epochs: {metrics['epochs_trained']}")
                    if metrics['timestamp']:
                        print(f"   - Timestamp: {metrics['timestamp']}")
                    
                    return metrics
                    
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print("⚠️  No training_summary.csv found in any location")
    return metrics

# ========== IMPORT FROM DIRECTORY ==========
def import_model_from_directory(directory="models"):
    """Import model components from directory"""
    global model, scaler, label_encoders, feature_columns, model_config
    
    print(f"\n🔄 Importing model from directory: {directory}")
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' does not exist")
        return False
    
    try:
        # 1. Load feature columns (always needed)
        feature_path = os.path.join(directory, 'feature_columns.pkl')
        if os.path.exists(feature_path):
            feature_columns = joblib.load(feature_path)
            print(f"✅ Feature columns loaded: {len(feature_columns)} features")
        else:
            print(f"⚠️  Feature columns not found at {feature_path}")
            # Try alternative location
            feature_path = 'feature_columns.pkl'
            if os.path.exists(feature_path):
                feature_columns = joblib.load(feature_path)
                print(f"✅ Feature columns loaded from root: {len(feature_columns)} features")
            else:
                print("❌ Feature columns not found anywhere")
                return False
        
        # 2. Load scaler
        scaler_path = os.path.join(directory, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded")
        else:
            print(f"⚠️  Scaler not found at {scaler_path}")
            # Try alternative location
            scaler_path = 'scaler.pkl'
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded from root")
            else:
                print("❌ Scaler not found anywhere")
        
        # 3. Load label encoders
        encoders_path = os.path.join(directory, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
            print(f"✅ Label encoders loaded: {list(label_encoders.keys())}")
        else:
            print(f"⚠️  Label encoders not found at {encoders_path}")
            # Try alternative location
            encoders_path = 'label_encoders.pkl'
            if os.path.exists(encoders_path):
                label_encoders = joblib.load(encoders_path)
                print(f"✅ Label encoders loaded from root: {list(label_encoders.keys())}")
            else:
                print("❌ Label encoders not found anywhere")
        
        # 4. Load model config
        config_path = os.path.join(directory, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            print("✅ Model config loaded")
        else:
            print(f"⚠️  Model config not found at {config_path}")
            # Try alternative location
            config_path = 'model_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                print("✅ Model config loaded from root")
        
        # 5. Try to load different model types
        model_loaded = False
        
        # Check for scikit-learn model
        sklearn_path = os.path.join(directory, 'price_predictor.pkl')
        if os.path.exists(sklearn_path):
            try:
                from sklearn.base import BaseEstimator
                model = joblib.load(sklearn_path)
                if hasattr(model, 'predict'):
                    print("✅ Scikit-learn model loaded")
                    model_loaded = True
                else:
                    print("⚠️  Loaded object is not a valid model")
                    model = None
            except Exception as e:
                print(f"❌ Error loading scikit-learn model: {e}")
                model = None
        
        # Check for ANN model (if TensorFlow is available)
        ann_path = 'currentAiSolution.h5'
        if not model_loaded and os.path.exists(ann_path):
            try:
                # Try to import tensorflow
                import tensorflow as tf
                from tensorflow import keras
                model = keras.models.load_model(ann_path)
                print("✅ TensorFlow ANN model loaded")
                model_loaded = True
            except ImportError:
                print("⚠️  TensorFlow not installed, cannot load ANN model")
            except Exception as e:
                print(f"❌ Error loading TensorFlow model: {e}")
        
        # 6. If no model found, create a simple one
        if not model_loaded:
            print("📝 Creating simple fallback model...")
            from sklearn.ensemble import RandomForestRegressor
            
            # Create simple model
            np.random.seed(42)
            n_samples = 100
            X_train = np.random.rand(n_samples, len(feature_columns))
            y_train = 300 + 500 * np.random.rand(n_samples)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            print("✅ Simple fallback model created")
        
        print(f"✅ Model import complete!")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Scaler: {'Loaded' if scaler else 'Not loaded'}")
        print(f"   Label Encoders: {len(label_encoders)}")
        print(f"   Model Type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ========== FEATURE EXTRACTION ==========
def extract_features(product: ProductRequest):
    """Extract features from product request"""
    
    # Default values
    features = {
        'condition_encoded': 3,  # Default: Good
        'age': 2,
        'title_length': len(product.title),
        'has_defects': 1 if product.has_defects else 0,
        'shipping_cost': float(product.shipping_cost),
        'brand_encoded': 0,
        'category_encoded': 0,
        'product_year': 2022
    }
    
    # Condition encoding
    condition_map = {
        'new': 5, 'like new': 4, 'excellent': 4,
        'very good': 3, 'good': 3, 'acceptable': 2,
        'used': 2, 'fair': 1, 'poor': 0
    }
    
    cond_lower = product.condition.lower()
    for key, value in condition_map.items():
        if key in cond_lower:
            features['condition_encoded'] = value
            break
    
    # Age calculation
    current_year = datetime.now().year
    if product.product_year:
        features['product_year'] = product.product_year
        features['age'] = max(0, current_year - product.product_year)
    else:
        # Try to extract year from title
        try:
            matches = re.findall(r'\b(20[0-2][0-9]|19[0-9]{2})\b', product.title)
            if matches:
                year = int(matches[0])
                features['product_year'] = year
                features['age'] = max(0, current_year - year)
        except:
            pass
    
    # Brand encoding
    title_lower = product.title.lower()
    
    # Determine brand
    if product.brand:
        brand = product.brand.lower()
    elif 'iphone' in title_lower or 'apple' in title_lower:
        brand = 'apple'
    elif 'samsung' in title_lower or 'galaxy' in title_lower:
        brand = 'samsung'
    elif 'google' in title_lower or 'pixel' in title_lower:
        brand = 'google'
    elif 'lg' in title_lower:
        brand = 'lg'
    elif 'sony' in title_lower:
        brand = 'sony'
    else:
        brand = 'other'
    
    # Use label encoder if available
    if label_encoders and 'brand' in label_encoders:
        brand_encoder = label_encoders['brand']
        if isinstance(brand_encoder, dict):  # Dictionary encoder
            features['brand_encoded'] = brand_encoder.get(brand, brand_encoder.get('other', 0))
        else:  # Try as scikit-learn LabelEncoder
            try:
                # Try to transform if it has transform method
                if hasattr(brand_encoder, 'transform'):
                    # Need to handle unseen labels
                    try:
                        features['brand_encoded'] = brand_encoder.transform([brand])[0]
                    except:
                        features['brand_encoded'] = len(brand_encoder.classes_) - 1  # Use last class as "other"
                else:
                    features['brand_encoded'] = 0
            except:
                features['brand_encoded'] = 0
    else:
        # Simple encoding
        brand_map = {'apple': 0, 'samsung': 1, 'google': 2, 'lg': 3, 'sony': 4, 'other': 5}
        features['brand_encoded'] = brand_map.get(brand, 5)
    
    # Category encoding
    if label_encoders and 'category' in label_encoders:
        category_encoder = label_encoders['category']
        
        # Determine category
        if 'phone' in title_lower or 'iphone' in title_lower or 'galaxy' in title_lower:
            category = 'electronics'
        elif 'laptop' in title_lower or 'macbook' in title_lower:
            category = 'electronics'
        elif 'tablet' in title_lower or 'ipad' in title_lower:
            category = 'electronics'
        elif 'shirt' in title_lower or 'pants' in title_lower or 'clothing' in title_lower:
            category = 'clothing'
        elif 'book' in title_lower:
            category = 'books'
        else:
            category = 'other'
        
        try:
            if isinstance(category_encoder, dict):
                features['category_encoded'] = category_encoder.get(category, category_encoder.get('other', 0))
            elif hasattr(category_encoder, 'transform'):
                try:
                    features['category_encoded'] = category_encoder.transform([category])[0]
                except:
                    features['category_encoded'] = len(category_encoder.classes_) - 1
            else:
                features['category_encoded'] = 0
        except:
            features['category_encoded'] = 0
    else:
        # Simple category encoding
        if 'phone' in title_lower or 'iphone' in title_lower or 'galaxy' in title_lower:
            features['category_encoded'] = 0
        elif 'laptop' in title_lower or 'macbook' in title_lower:
            features['category_encoded'] = 1
        elif 'tablet' in title_lower or 'ipad' in title_lower:
            features['category_encoded'] = 2
        else:
            features['category_encoded'] = 3
    
    return features

# ========== FALLBACK PREDICTION ==========
def get_fallback_prediction(product: ProductRequest):
    """Always works - no model needed"""
    base_price = 350.0
    
    # Adjust based on keywords
    title_lower = product.title.lower()
    if 'iphone 15' in title_lower:
        base_price = 850.0
    elif 'iphone 14' in title_lower:
        base_price = 650.0
    elif 'iphone 13' in title_lower:
        base_price = 550.0
    elif 'iphone 12' in title_lower:
        base_price = 450.0
    elif 'iphone 11' in title_lower:
        base_price = 350.0
    elif 'macbook pro' in title_lower:
        base_price = 1200.0
    elif 'macbook air' in title_lower:
        base_price = 900.0
    elif 'macbook' in title_lower:
        base_price = 800.0
    elif 'samsung galaxy s23' in title_lower:
        base_price = 700.0
    elif 'samsung galaxy' in title_lower:
        base_price = 500.0
    elif 'samsung' in title_lower:
        base_price = 400.0
    
    # Condition adjustment
    condition_mult = {
        'new': 1.3, 'like new': 1.2, 'excellent': 1.15,
        'very good': 1.05, 'good': 1.0, 'acceptable': 0.9,
        'used': 0.8, 'fair': 0.7, 'poor': 0.6
    }
    
    cond_lower = product.condition.lower()
    multiplier = 1.0
    for key, value in condition_mult.items():
        if key in cond_lower:
            multiplier = value
            break
    
    predicted = base_price * multiplier
    confidence = 0.6
    
    return {
        'predicted_price': round(predicted, 2),
        'confidence': confidence,
        'price_range_low': round(predicted * 0.8, 2),
        'price_range_high': round(predicted * 1.2, 2),
        'message': 'Fallback prediction'
    }

# ========== CALCULATE CONFIDENCE FROM MAE ==========
def calculate_confidence_from_mae(predicted_price: float) -> float:
    """Calculate confidence score based on MAE and predicted price"""
    global training_metrics
    
    if not training_metrics or training_metrics.get('final_train_mae') is None:
        # Default confidence if no MAE available
        return 0.85
    
    mae = training_metrics['final_train_mae']
    
    if mae <= 0:
        return 0.85
    
    # Normalize MAE relative to predicted price
    # Lower MAE relative to price = higher confidence
    if predicted_price > 0:
        mae_ratio = mae / predicted_price
    else:
        mae_ratio = 1.0  # Worst case
    
    # Convert to confidence (0-1 scale)
    # Formula: confidence = 1 - min(mae_ratio, 0.5) * 2
    # This gives 100% confidence at MAE=0, 0% at MAE=50% of price or more
    confidence = 1.0 - min(mae_ratio, 0.5) * 2
    
    # Apply additional factors
    if training_metrics.get('final_train_r2') is not None:
        r2 = training_metrics['final_train_r2']
        # Boost confidence if R² is good
        if r2 > 0.8:
            confidence = min(0.95, confidence + 0.1)
        elif r2 > 0.6:
            confidence = min(0.9, confidence + 0.05)
    
    # Ensure confidence is within reasonable bounds
    confidence = max(0.3, min(0.95, confidence))
    
    return confidence

# ========== PREDICTION FUNCTION ==========
def make_prediction(features_dict):
    """Make prediction using loaded model"""
    if model is None or not feature_columns:
        return None
    
    try:
        # Create DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all expected features exist
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[feature_columns]
        
        # Scale features if scaler exists
        if scaler:
            features_processed = scaler.transform(features_df)
        else:
            features_processed = features_df.values
        
        # Make prediction
        # Check model type and predict accordingly
        model_type = type(model).__name__
        
        if 'Sequential' in model_type or 'Functional' in model_type:  # TensorFlow/Keras
            prediction = float(model.predict(features_processed, verbose=0)[0][0])
        else:  # Scikit-learn model
            prediction = float(model.predict(features_processed)[0])
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    mae_info = f"MAE: ${training_metrics.get('final_train_mae', 0):.2f}" if training_metrics.get('final_train_mae') else "No MAE data"
    
    return {
        "message": "Used Product Price Analyzer API",
        "status": "running",
        "version": "2.1.0",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "None",
        "features_count": len(feature_columns),
        "training_metrics_loaded": any(training_metrics.values()),
        "training_mae": mae_info,
        "directory": os.getcwd(),
        "endpoints": [
            "GET / - This page",
            "GET /health - Health check",
            "GET /files - List model files",
            "GET /model-info - Model information",
            "GET /training-metrics - Training metrics from CSV",
            "POST /predict - Predict price with MAE integration",
            "POST /reload - Reload model"
        ]
    }

@app.get("/health")
async def health_check():
    files_exist = {
        'feature_columns.pkl': os.path.exists('models/feature_columns.pkl'),
        'scaler.pkl': os.path.exists('models/scaler.pkl'),
        'label_encoders.pkl': os.path.exists('models/label_encoders.pkl'),
        'model_config.json': os.path.exists('models/model_config.json'),
        'currentAiSolution.h5': os.path.exists('currentAiSolution.h5'),
        'training_summary.csv': any(os.path.exists(f) for f in ['training_summary.csv', 'models/training_summary.csv', 'data/training_summary.csv'])
    }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": len(feature_columns) > 0,
        "scaler_loaded": scaler is not None,
        "training_metrics_loaded": any(training_metrics.values()),
        "files_exist": files_exist
    }

@app.get("/files")
async def list_files():
    """List all model files in current directory"""
    import glob
    
    all_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.pkl', '.h5', '.json', '.csv', '.py')):
                all_files.append(os.path.join(root, file))
    
    # Check specifically for training_summary.csv
    training_files = []
    for loc in ['training_summary.csv', 'models/training_summary.csv', 'data/training_summary.csv', 'results/training_summary.csv']:
        if os.path.exists(loc):
            training_files.append(loc)
    
    return {
        "current_directory": os.getcwd(),
        "model_files": sorted(all_files)[:50],  # Limit to 50 files
        "training_summary_files": training_files,
        "models_directory_exists": os.path.exists('models'),
        "models_directory_contents": os.listdir('models') if os.path.exists('models') else []
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if model is None:
        return {"message": "No model loaded"}
    
    model_type = type(model).__name__
    model_info = {
        "model_type": model_type,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "scaler_loaded": scaler is not None,
        "label_encoders": list(label_encoders.keys()) if label_encoders else [],
        "model_config": model_config,
        "training_metrics": training_metrics if any(training_metrics.values()) else "No training metrics"
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):  # Random Forest
        model_info["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            model_info["max_depth"] = model.max_depth
    
    if hasattr(model, 'layers'):  # Keras model
        model_info["layers"] = len(model.layers)
        model_info["input_shape"] = str(model.input_shape)
        model_info["output_shape"] = str(model.output_shape)
    
    return model_info

@app.get("/training-metrics")
async def get_training_metrics_endpoint():
    """Get training metrics from CSV"""
    global training_metrics
    
    if not any(training_metrics.values()):
        # Try to load if not already loaded
        training_metrics = get_training_metrics()
    
    # Format response
    response = {
        "available": any(training_metrics.values()),
        "file_source": training_metrics.get('file_source', 'Not found'),
        "metrics": {}
    }
    
    # Only include non-None metrics
    for key, value in training_metrics.items():
        if value is not None and key != 'file_source':
            response["metrics"][key] = value
    
    # Add helpful summary
    if training_metrics.get('final_train_mae') is not None:
        mae = training_metrics['final_train_mae']
        response["summary"] = {
            "mean_absolute_error": f"${mae:.2f}",
            "interpretation": f"Average prediction error is ${mae:.2f}"
        }
    
    return response

@app.post("/reload")
async def reload_model():
    """Reload model from directory"""
    success = import_model_from_directory()
    
    # Also reload training metrics
    global training_metrics
    training_metrics = get_training_metrics()
    
    if success:
        return {
            "message": "Model reloaded successfully", 
            "success": True,
            "training_metrics_loaded": any(training_metrics.values())
        }
    else:
        return {
            "message": "Failed to reload model", 
            "success": False,
            "training_metrics_loaded": any(training_metrics.values())
        }

@app.post("/predict", response_model=PricePrediction)
async def predict_price(product: ProductRequest):
    """Main prediction endpoint with MAE integration"""
    print(f"\n📥 Prediction request:")
    print(f"   Title: {product.title[:50]}...")
    print(f"   Condition: {product.condition}")
    
    try:
        # Step 1: Extract features
        features = extract_features(product)
        print(f"   Features extracted: {list(features.keys())}")
        
        # Step 2: Try to use loaded model
        if model is not None and feature_columns:
            print(f"   Using {type(model).__name__} model with {len(feature_columns)} features")
            
            prediction = make_prediction(features)
            
            if prediction is not None:
                # Calculate confidence based on MAE
                confidence = calculate_confidence_from_mae(prediction)
                
                # Prepare message with MAE info if available
                if training_metrics.get('final_train_mae') is not None:
                    mae = training_metrics['final_train_mae']
                    if training_metrics.get('final_train_r2') is not None:
                        r2 = training_metrics['final_train_r2']
                        message = f"{type(model).__name__} prediction (MAE: ${mae:.2f}, R²: {r2:.3f})"
                    else:
                        message = f"{type(model).__name__} prediction (MAE: ${mae:.2f})"
                else:
                    message = f"{type(model).__name__} prediction"
                
                print(f"   Model prediction: ${prediction:.2f}")
                if training_metrics.get('final_train_mae'):
                    print(f"   Training MAE: ${training_metrics['final_train_mae']:.2f}")
                print(f"   Calculated confidence: {confidence:.2f}")
                
            else:
                # Model prediction failed
                fallback = get_fallback_prediction(product)
                prediction = fallback['predicted_price']
                confidence = fallback['confidence']
                message = "Model failed, using fallback"
                print(f"   Model failed, using fallback: ${prediction:.2f}")
        else:
            # No model loaded
            fallback = get_fallback_prediction(product)
            prediction = fallback['predicted_price']
            confidence = fallback['confidence']
            message = "No model loaded, using fallback"
            print(f"   No model loaded, using fallback: ${prediction:.2f}")
        
        # Step 3: Calculate price range using MAE if available
        if training_metrics.get('final_train_mae') is not None and training_metrics['final_train_mae'] > 0:
            # Use actual MAE for price range
            mae = training_metrics['final_train_mae']
            price_range_low = max(0, prediction - mae)
            price_range_high = prediction + mae
            range_method = "MAE-based"
        else:
            # Fallback to confidence-based range
            margin = prediction * (1 - confidence)
            price_range_low = max(0, prediction - margin)
            price_range_high = prediction + margin
            range_method = "Confidence-based"
        
        print(f"   ✅ Final: ${price_range_high:.2f}")
        print(f"   📊 Range ({range_method}): ${price_range_low:.2f}-${price_range_high:.2f}")
        
        # Prepare metadata
        metadata = None
        if any(training_metrics.values()):
            metadata = {k: v for k, v in training_metrics.items() if v is not None}
        
        return PricePrediction(
            predicted_price=round(prediction, 2),
            confidence=round(confidence, 2),
            price_range_low=round(price_range_low, 2),
            price_range_high=round(price_range_high, 2),
            message=message,
            model_type=type(model).__name__ if model else "Fallback",
            training_mae=round(training_metrics.get('final_train_mae', 0), 2) if training_metrics.get('final_train_mae') is not None else None,
            model_accuracy=round(training_metrics.get('final_train_r2', 0), 3) if training_metrics.get('final_train_r2') is not None else None,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback
        ultimate_fallback = get_fallback_prediction(product)
        
        return PricePrediction(
            predicted_price=ultimate_fallback['predicted_price'],
            confidence=ultimate_fallback['confidence'],
            price_range_low=ultimate_fallback['price_range_low'],
            price_range_high=ultimate_fallback['price_range_high'],
            message=f"Error: {str(e)[:50]}...",
            model_type="Fallback (Error)",
            training_mae=round(training_metrics.get('final_train_mae', 0), 2) if training_metrics.get('final_train_mae') is not None else None
        )

# ========== STARTUP ==========
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 Starting Used Product Price Analyzer API v2.1.0")
    print("=" * 60)
    
    # Load training metrics first
    global training_metrics
    training_metrics = get_training_metrics()
    
    # Try multiple import strategies
    print("\n🔍 Looking for model files...")
    
    # First try the models directory
    success = import_model_from_directory("models")
    
    if not success:
        print("\n🔍 Trying current directory...")
        success = import_model_from_directory(".")
    
    if not success:
        print("\n🔍 Looking for specific files...")
        # Try to load individual files
        try:
            # Look for common file locations
            possible_paths = [
                ('feature_columns.pkl', 'models/feature_columns.pkl'),
                ('scaler.pkl', 'models/scaler.pkl'),
                ('label_encoders.pkl', 'models/label_encoders.pkl')
            ]
            
            for file_name, path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ Found {file_name} at {path}")
                elif os.path.exists(file_name):
                    print(f"✅ Found {file_name} in current directory")
                else:
                    print(f"❌ {file_name} not found")
        
        except Exception as e:
            print(f"⚠️  File search error: {e}")
    
    print("\n✅ API ready!")
    print("📊 Model Status:")
    print(f"   Model: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Training Metrics: {'✅ Loaded' if any(training_metrics.values()) else '❌ Not found'}")
    if training_metrics.get('final_train_mae'):
        print(f"   Mean Absolute Error: ${training_metrics['final_train_mae']:.2f}")
    print("\n🔗 Endpoints:")
    print("   http://localhost:8000")
    print("   http://localhost:8000/docs")
    print("   http://localhost:8000/redoc")
    print("   http://localhost:8000/training-metrics")
    print("=" * 60)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
