# main.py - Complete ANN Price Predictor with Scraped Data Integration
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# Import our ANN predictor
try:
    from price_predictor import PricePredictorANN
    print("✅ Imported PricePredictorANN from price_predictor.py")
except ImportError as e:
    print(f"❌ Failed to import PricePredictorANN: {e}")
    print("   Make sure price_predictor.py is in the same directory.")
    exit(1)

def create_sample_data():
    """
    Load or create data for training the ANN model
    This function tries multiple data sources in order:
    1. Existing CSV files with scraped data
    2. Generated synthetic data as fallback
    """
    print("\n📊 Loading training data...")
    
    # List of possible scraped data files to try
    scraped_data_files = [
        'data/training_data_normalized.csv',
        'data/scraped_products.csv', 
        'data/ebay_listings.csv',
        'data/amazon_products.csv',
        'scraped_data.csv',
        'products_data.csv',
        'training_data.csv'
    ]
    
    # Try each file
    for file_path in scraped_data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Loaded {len(df)} rows from {file_path}")
                
                # Basic validation
                if len(df) < 5:
                    print(f"⚠️  File has only {len(df)} rows, trying next source...")
                    continue
                    
                if 'price' not in df.columns:
                    print(f"⚠️  No 'price' column found in {file_path}")
                    # Try to find price column with different names
                    price_columns = [col for col in df.columns if 'price' in col.lower() or 'cost' in col.lower()]
                    if price_columns:
                        df['price'] = df[price_columns[0]]
                        print(f"   Using '{price_columns[0]}' as price column")
                    else:
                        print(f"   Cannot find price data, trying next source...")
                        continue
                
                print(f"   Columns: {df.columns.tolist()}")
                print(f"   Price stats - Min: ${df['price'].min():.2f}, Max: ${df['price'].max():.2f}, Mean: ${df['price'].mean():.2f}")
                return df
                
            except Exception as e:
                print(f"⚠️  Could not read {file_path}: {e}")
                continue
    
    # If no scraped data found, create synthetic data
    print("⚠️  No scraped data files found. Creating synthetic training data...")
    return create_synthetic_data()

def create_synthetic_data():
    """Create realistic synthetic data for ANN training when no scraped data is available"""
    print("   Generating synthetic product data...")
    
    np.random.seed(42)
    n_samples = 2000  # Good amount for ANN training
    
    # Create realistic product data
    data = {
        'title': [f'Used Product {i}' for i in range(n_samples)],
        'condition': np.random.choice(['New', 'Like New', 'Good', 'Fair', 'Poor'], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'brand': np.random.choice(['Apple', 'Samsung', 'Sony', 'Dell', 'HP', 'Nike', 'Adidas', 'Generic'], n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Toys'], n_samples),
        'age': np.random.exponential(3, n_samples).astype(int) + 1,  # 1-10 years
        'has_defects': np.random.binomial(1, 0.25, n_samples),  # 25% have defects
        'shipping': np.random.uniform(0, 50, n_samples),
        'year': np.random.randint(2015, 2024, n_samples),  # Manufacture year
        'description': [f'Product description {i}' for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic prices based on features
    base_prices = {
        'Electronics': 300,
        'Clothing': 50,
        'Home': 150,
        'Books': 20,
        'Sports': 100,
        'Toys': 30
    }
    
    condition_multipliers = {
        'New': 1.0,
        'Like New': 0.8,
        'Good': 0.6,
        'Fair': 0.4,
        'Poor': 0.2
    }
    
    brand_multipliers = {
        'Apple': 1.5,
        'Samsung': 1.3,
        'Sony': 1.2,
        'Dell': 1.1,
        'HP': 1.1,
        'Nike': 1.2,
        'Adidas': 1.1,
        'Generic': 0.8
    }
    
    # Calculate price for each product
    prices = []
    for idx, row in df.iterrows():
        base = base_prices.get(row['category'], 100)
        condition_mult = condition_multipliers.get(row['condition'], 0.5)
        brand_mult = brand_multipliers.get(row['brand'], 1.0)
        age_factor = 1.0 - (row['age'] * 0.05)  # 5% reduction per year
        defect_factor = 0.7 if row['has_defects'] else 1.0
        
        price = base * condition_mult * brand_mult * age_factor * defect_factor
        price += np.random.normal(0, price * 0.1)  # Add some randomness
        price = max(10, price)  # Minimum $10
        
        prices.append(price)
    
    df['price'] = prices
    df['price'] = df['price'].round(2)
    
    print(f"✅ Created synthetic data with {n_samples} products")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"   Average price: ${df['price'].mean():.2f}")
    
    # Save the synthetic data for future use
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_training_data.csv', index=False)
    print("💾 Saved synthetic data to data/synthetic_training_data.csv")
    
    return df

def prepare_data_for_ann(df, predictor):
    """
    Prepare scraped/synthetic data for ANN training
    Converts data to the format expected by PricePredictorANN
    """
    print("\n🔧 Preparing data for ANN training...")
    
    if df is None or len(df) == 0:
        print("❌ No data to prepare")
        return None, None
    
    df_processed = df.copy()
    
    # Ensure we have the required columns
    print("   Checking and creating required features...")
    
    # 1. Title length (if title exists)
    if 'title' in df_processed.columns:
        df_processed['title_length'] = df_processed['title'].apply(len)
        print(f"   ✓ Created 'title_length' from title column")
    elif 'description' in df_processed.columns:
        df_processed['title_length'] = df_processed['description'].apply(len)
        print(f"   ✓ Created 'title_length' from description column")
    else:
        df_processed['title_length'] = 50  # Default
        print(f"   ⚠️  Created default 'title_length' (50)")
    
    # 2. Condition encoding (convert text to numeric if needed)
    if 'condition' in df_processed.columns:
        condition_mapping = {
            'New': 5, 'Like New': 4, 'Good': 3, 'Fair': 2, 'Poor': 1,
            'Excellent': 5, 'Very Good': 4, 'Acceptable': 2, 'Bad': 1
        }
        # Convert text conditions to numeric
        df_processed['condition_encoded'] = df_processed['condition'].apply(
            lambda x: condition_mapping.get(str(x).title(), 3)
        )
        print(f"   ✓ Encoded 'condition' to numeric values")
    else:
        df_processed['condition_encoded'] = 3  # Default "Good"
        print(f"   ⚠️  Created default 'condition_encoded' (3 - Good)")
    
    # 3. Age (if not present, create from year or use default)
    if 'age' not in df_processed.columns:
        if 'year' in df_processed.columns:
            current_year = 2024
            df_processed['age'] = current_year - df_processed['year']
            print(f"   ✓ Calculated 'age' from year column")
        else:
            df_processed['age'] = 2  # Default 2 years
            print(f"   ⚠️  Created default 'age' (2 years)")
    
    # 4. Has defects (ensure binary)
    if 'has_defects' in df_processed.columns:
        df_processed['has_defects'] = df_processed['has_defects'].apply(lambda x: 1 if x else 0)
        print(f"   ✓ Converted 'has_defects' to binary")
    else:
        # Check for defect-related columns
        defect_cols = [col for col in df_processed.columns if 'defect' in col.lower() or 'damage' in col.lower()]
        if defect_cols:
            df_processed['has_defects'] = df_processed[defect_cols[0]].apply(lambda x: 1 if x else 0)
            print(f"   ✓ Created 'has_defects' from '{defect_cols[0]}' column")
        else:
            df_processed['has_defects'] = 0  # Default no defects
            print(f"   ⚠️  Created default 'has_defects' (0 - no defects)")
    
    # 5. Shipping cost
    if 'shipping' in df_processed.columns:
        df_processed['shipping_cost'] = df_processed['shipping'].astype(float)
        print(f"   ✓ Using 'shipping' as shipping_cost")
    elif 'shipping_cost' in df_processed.columns:
        df_processed['shipping_cost'] = df_processed['shipping_cost'].astype(float)
        print(f"   ✓ Using existing 'shipping_cost' column")
    else:
        df_processed['shipping_cost'] = 0.0  # Default free shipping
        print(f"   ⚠️  Created default 'shipping_cost' (0 - free shipping)")
    
    # 6. Product year
    if 'year' in df_processed.columns:
        df_processed['product_year'] = df_processed['year'].astype(int)
        print(f"   ✓ Using 'year' as product_year")
    elif 'product_year' in df_processed.columns:
        df_processed['product_year'] = df_processed['product_year'].astype(int)
        print(f"   ✓ Using existing 'product_year' column")
    else:
        df_processed['product_year'] = 2022  # Default year
        print(f"   ⚠️  Created default 'product_year' (2022)")
    
    # 7. Brand and Category (will be encoded by prepare_features)
    if 'brand' not in df_processed.columns:
        df_processed['brand'] = 'Unknown'
        print(f"   ⚠️  Created default 'brand' (Unknown)")
    
    if 'category' not in df_processed.columns:
        df_processed['category'] = 'General'
        print(f"   ⚠️  Created default 'category' (General)")
    
    # 8. Price validation
    if 'price' not in df_processed.columns:
        print("❌ CRITICAL: No 'price' column found!")
        print("   Cannot train model without target variable.")
        return None, None
    
    # Ensure price is numeric
    df_processed['price'] = pd.to_numeric(df_processed['price'], errors='coerce')
    
    # Remove rows with invalid prices
    initial_count = len(df_processed)
    df_processed = df_processed.dropna(subset=['price'])
    df_processed = df_processed[df_processed['price'] > 0]
    
    if len(df_processed) < initial_count:
        print(f"   ⚠️  Removed {initial_count - len(df_processed)} rows with invalid prices")
    
    print(f"   Processed {len(df_processed)} valid samples")
    
    # Use the predictor's prepare_features method
    try:
        X, y = predictor.prepare_features(df_processed)
        print(f"✅ Successfully prepared {len(X)} samples with {X.shape[1]} features")
        return X, y
    except Exception as e:
        print(f"❌ Failed to prepare features: {e}")
        return None, None

def train_and_evaluate_ann(predictor, X_train, y_train, X_val, y_val):
    """Train the ANN model and return evaluation results"""
    print("\n🧠 Training Artificial Neural Network...")
    print("   This may take a moment...")
    
    start_time = time.time()
    
    try:
        # Train the model
        history = predictor.train_model(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        # Extract metrics
        train_metrics = history.get('train_metrics', {})
        val_metrics = history.get('val_metrics', {})
        
        print(f"✅ ANN training completed in {training_time:.1f} seconds")
        print("\n📊 Training Results:")
        print(f"   Final Training Loss: {train_metrics.get('loss', 0):.4f}")
        print(f"   Final Training MAE: ${train_metrics.get('mae', 0):.2f}")
        print(f"   Final Training R²: {train_metrics.get('r2', 0):.3f}")
        print(f"   Final Validation MAE: ${val_metrics.get('mae', 0):.2f}")
        print(f"   Final Validation R²: {val_metrics.get('r2', 0):.3f}")
        
        return history
        
    except Exception as e:
        print(f"❌ ANN training failed: {e}")
        
        # Try simpler model as fallback
        print("🔄 Attempting with simpler model configuration...")
        predictor.epochs = 100
        predictor.hidden_units = [32, 16]
        
        try:
            history = predictor.train_model(X_train, y_train, X_val, y_val)
            print("✅ Training completed with simplified configuration")
            return history
        except Exception as e2:
            print(f"❌ Simplified training also failed: {e2}")
            return None

def make_example_predictions(predictor, df_sample=None):
    """Make example predictions to demonstrate the model"""
    print("\n🔮 Making example predictions...")
    
    examples = []
    
    # Example 1: High-end electronics in good condition
    example1 = {
        'condition_encoded': 4,  # Like New
        'age': 1,
        'title_length': 45,
        'has_defects': 0,
        'shipping_cost': 15.99,
        'brand_encoded': 0,  # Will be set by predictor
        'category_encoded': 0,  # Will be set by predictor
        'product_year': 2023
    }
    
    # Example 2: Older item with defects
    example2 = {
        'condition_encoded': 2,  # Fair
        'age': 5,
        'title_length': 35,
        'has_defects': 1,
        'shipping_cost': 25.50,
        'brand_encoded': 0,
        'category_encoded': 0,
        'product_year': 2018
    }
    
    # Example 3: Average item
    example3 = {
        'condition_encoded': 3,  # Good
        'age': 3,
        'title_length': 40,
        'has_defects': 0,
        'shipping_cost': 0.0,
        'brand_encoded': 0,
        'category_encoded': 0,
        'product_year': 2021
    }
    
    examples = [example1, example2, example3]
    
    for i, example in enumerate(examples, 1):
        try:
            prediction = predictor.predict_price(example)
            if prediction:
                print(f"\n   Example {i}:")
                condition_text = {5: 'New', 4: 'Like New', 3: 'Good', 2: 'Fair', 1: 'Poor'}.get(example['condition_encoded'], 'Unknown')
                print(f"     Condition: {condition_text} ({example['condition_encoded']}/5)")
                print(f"     Age: {example['age']} years")
                print(f"     Defects: {'Yes' if example['has_defects'] else 'No'}")
                print(f"     Shipping: ${example['shipping_cost']:.2f}")
                print(f"     Predicted Price: ${prediction['predicted_price']:,.2f}")
                print(f"     Price Range: ${prediction['price_range']['low']:,.2f} - ${prediction['price_range']['high']:,.2f}")
                print(f"     Confidence: {prediction['confidence_score']:.1%}")
        except Exception as e:
            print(f"   ❌ Could not make prediction for example {i}: {e}")
    
    # If we have sample data, make a prediction on a real sample
    if df_sample is not None and len(df_sample) > 0:
        print("\n📈 Predicting price for a real sample from the data...")
        try:
            # Take the first row as example
            sample_row = df_sample.iloc[0]
            
            # Create features from sample
            sample_features = {}
            if hasattr(predictor, 'feature_columns'):
                for feature in predictor.feature_columns:
                    if feature in df_sample.columns:
                        sample_features[feature] = sample_row[feature]
                    else:
                        # Use default values for missing features
                        if 'condition' in feature:
                            sample_features[feature] = 3
                        elif 'age' in feature:
                            sample_features[feature] = 2
                        elif 'title' in feature:
                            sample_features[feature] = 40
                        elif 'defects' in feature:
                            sample_features[feature] = 0
                        elif 'shipping' in feature:
                            sample_features[feature] = 0
                        elif 'brand' in feature:
                            sample_features[feature] = 0
                        elif 'category' in feature:
                            sample_features[feature] = 0
                        elif 'year' in feature:
                            sample_features[feature] = 2022
                        else:
                            sample_features[feature] = 0
            
            sample_prediction = predictor.predict_price(sample_features)
            if sample_prediction:
                actual_price = sample_row['price'] if 'price' in sample_row else 'N/A'
                print(f"\n   Real Sample Prediction:")
                print(f"     Actual Price: ${actual_price}" if actual_price != 'N/A' else "     Actual Price: Unknown")
                print(f"     Predicted Price: ${sample_prediction['predicted_price']:,.2f}")
                if actual_price != 'N/A':
                    error = abs(actual_price - sample_prediction['predicted_price'])
                    error_percent = (error / actual_price) * 100
                    print(f"     Prediction Error: ${error:.2f} ({error_percent:.1f}%)")
        except Exception as e:
            print(f"   ⚠️  Could not predict sample: {e}")

def main():
    """Main training pipeline for ANN Price Predictor"""
    print("=" * 70)
    print("🚀 ARTIFICIAL NEURAL NETWORK PRICE PREDICTOR TRAINING")
    print("=" * 70)
    os.system("python tryy.py")

    total_start_time = time.time()
    
    # Step 1: Load or create data
    print("\n1️⃣ DATA COLLECTION")
    print("-" * 50)
    df = create_sample_data()
    
    if df is None or len(df) == 0:
        print("❌ No data available for training. Exiting...")
        return None
    
    print(f"✅ Loaded {len(df)} product listings")
    
    # Step 2: Initialize ANN Predictor
    print("\n2️⃣ MODEL INITIALIZATION")
    print("-" * 50)
    predictor = PricePredictorANN()
    print(f"✅ Initialized PricePredictorANN")
    print(f"   Architecture: {predictor.input_dim}-{'-'.join(map(str, predictor.hidden_units))}-1")
    print(f"   Learning Rate: {predictor.learning_rate}")
    print(f"   Epochs: {predictor.epochs}")
    print(f"   Batch Size: {predictor.batch_size}")
    
    # Step 3: Prepare data for ANN
    print("\n3️⃣ DATA PREPARATION")
    print("-" * 50)
    X, y = prepare_data_for_ann(df, predictor)
    
    if X is None or y is None:
        print("❌ Data preparation failed. Exiting...")
        return None
    
    # Step 4: Split data
    print("\n4️⃣ DATA SPLITTING")
    print("-" * 50)
    
    if len(X) >= 100:
        test_size = 0.2
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        print(f"✅ Split: {len(X_train)} training, {len(X_val)} validation samples")
        print(f"   Training/Validation ratio: {100*(1-test_size):.0f}/{100*test_size:.0f}")
    elif len(X) >= 30:
        test_size = 0.15
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        print(f"⚠️  Small dataset: {len(X_train)} training, {len(X_val)} validation")
    elif len(X) >= 10:
        test_size = 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        print(f"⚠️  Very small dataset: {len(X_train)} training, {len(X_val)} validation")
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y
        print(f"⚠️  Minimal dataset: Using all {len(X_train)} samples for training")
    
    # Step 5: Train ANN model
    print("\n5️⃣ MODEL TRAINING")
    print("-" * 50)
    history = train_and_evaluate_ann(predictor, X_train, y_train, X_val, y_val)
    
    if history is None:
        print("❌ Model training failed. Exiting...")
        return None
    
    # Step 6: Save the trained model
    print("\n6️⃣ MODEL SAVING")
    print("-" * 50)
    try:
        predictor.save_model()
        print(f"✅ Model saved successfully!")
        print(f"   Main model: {predictor.model_path}")
        print(f"   Model config: {predictor.model_config_path}")
        print(f"   Training artifacts: {predictor.learning_base_path}")
        
        # List generated files
        if os.path.exists(predictor.learning_base_path):
            files = os.listdir(predictor.learning_base_path)
            print(f"   Generated {len(files)} training artifacts")
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        print("🔄 Attempting emergency save...")
        try:
            predictor.model.save('emergency_ann_model.h5')
            print("✅ Emergency model saved as 'emergency_ann_model.h5'")
        except:
            print("❌ Could not save model at all")
    
    # Step 7: Make example predictions
    print("\n7️⃣ EXAMPLE PREDICTIONS")
    print("-" * 50)
    make_example_predictions(predictor, df)
    
    # Step 8: Final summary
    print("\n8️⃣ TRAINING SUMMARY")
    print("-" * 50)
    
    total_time = time.time() - total_start_time
    
    print(f"📊 Data Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Features used: {predictor.input_dim}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Total training time: {total_time:.1f} seconds")
    
    print(f"\n💾 Model Files:")
    model_files = []
    if os.path.exists(predictor.model_path):
        model_files.append(predictor.model_path)
    if os.path.exists(predictor.model_config_path):
        model_files.append(predictor.model_config_path)
    if os.path.exists('models/'):
        model_files.extend([f'models/{f}' for f in os.listdir('models/')])
    
    for file in model_files[:5]:  # Show first 5 files
        print(f"   • {file}")
    
    if len(model_files) > 5:
        print(f"   • ... and {len(model_files) - 5} more files")
    
    print(f"\n📈 Next Steps:")
    print("   1. Check learningBase/ folder for training visualizations")
    print("   2. Use TensorBoard: tensorboard --logdir=learningBase/tensorboard_logs/")
    print("   3. Integrate model using predictor.load_model()")
    print("   4. Collect more data to improve accuracy")
    
    print("\n" + "=" * 70)
    print("✅ ANN PRICE PREDICTOR TRAINING COMPLETE!")
    print("=" * 70)
    
    return predictor

def quick_test():
    """Quick test function to verify the predictor works after training"""
    print("\n🧪 Quick Model Test")
    print("-" * 50)
    
    try:
        # Try to load a saved model
        if os.path.exists('currentAiSolution.h5'):
            print("Loading saved model...")
            predictor = PricePredictorANN()
            predictor.load_model()
            
            # Make a test prediction
            test_features = {
                'condition_encoded': 3,
                'age': 2,
                'title_length': 40,
                'has_defects': 0,
                'shipping_cost': 10.0,
                'brand_encoded': 0,
                'category_encoded': 0,
                'product_year': 2022
            }
            
            prediction = predictor.predict_price(test_features)
            if prediction:
                print(f"✅ Model loaded and working!")
                print(f"   Test prediction: ${prediction['predicted_price']:.2f}")
                return True
        else:
            print("⚠️  No saved model found. Run training first.")
            return False
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("learningBase", exist_ok=True)
    
    print("🔧 Setting up directories...")
    print(f"   Data directory: {'data/'} {'✓' if os.path.exists('data/') else '✗'}")
    print(f"   Models directory: {'models/'} {'✓' if os.path.exists('models/') else '✗'}")
    print(f"   Learning base: {'learningBase/'} {'✓' if os.path.exists('learningBase/') else '✗'}")
    
    # Run the main training pipeline
    predictor = main()
    
    if predictor:
        print("\n🎉 ANN Price Predictor is ready!")
        print("   Use: predictor.predict_price(features_dict) for predictions")
        print("   Use: predictor.load_model() to reload saved model")
        
        # Optional: Run quick test
        test = input("\nRun quick model test? (y/n): ")
        if test.lower() == 'y':
            quick_test()
    else:
        print("\n❌ Training failed. Check error messages above.")
# Save this as main.py
