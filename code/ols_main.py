# ols_main.py - Fixed Main Training Pipeline for OLS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
from datetime import datetime
import json

# Import OLS predictor
from ols_price_predictor import OLSPricePredictor

def create_sample_data():
    """Load data from CSV or create synthetic data"""
    # Try to load from CSV first
    csv_files = [
        'data/training_data_normalized.csv',
        'data/scraped_products.csv',
        'data/synthetic_training_data.csv',
        'training_data.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"✅ Loaded {len(df)} rows from {csv_file}")
                
                # Ensure price column exists
                if 'price' not in df.columns:
                    # Look for price-related columns
                    price_cols = [col for col in df.columns if 'price' in col.lower() or 'cost' in col.lower()]
                    if price_cols:
                        df['price'] = df[price_cols[0]]
                        print(f"   Using '{price_cols[0]}' as price column")
                    else:
                        print("   ❌ No price column found")
                        continue
                
                return df
            except Exception as e:
                print(f"⚠️  Could not read {csv_file}: {e}")
    
    # Create synthetic data if no files found
    print("⚠️  No data files found. Creating synthetic data...")
    return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data for OLS training"""
    np.random.seed(42)
    n_samples = 200
    
    # Create realistic product data
    data = {
        'condition': np.random.randint(1, 6, n_samples),
        'age': np.random.exponential(2, n_samples).astype(int) + 1,
        'title': ['Item ' + str(i) for i in range(n_samples)],
        'has_defects': np.random.binomial(1, 0.2, n_samples),
        'shipping': np.random.uniform(0, 50, n_samples),
        'brand': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'category': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'year': np.random.randint(2015, 2024, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['title_length'] = df['title'].apply(len)
    
    # Create price with clear linear relationships (good for OLS)
    df['price'] = (
        100 +  # Base price
        25 * df['condition'] +  # Better condition = higher price
        (-15) * df['age'] +  # Older = lower price
        (-40) * df['has_defects'] +  # Defects reduce price
        0.5 * df['shipping'] +  # Shipping affects price
        np.random.normal(0, 30, n_samples)  # Random noise
    )
    
    # Ensure positive prices
    df['price'] = df['price'].clip(lower=10)
    
    print(f"✅ Created {n_samples} synthetic samples for OLS training")
    return df

def prepare_data_for_ols(df, predictor):
    """Prepare data for OLS training"""
    print("\n🔧 Preparing data for OLS training...")
    
    if df is None or len(df) == 0:
        print("❌ No data to prepare")
        return None, None, None
    
    try:
        X, y, X_scaled = predictor.prepare_features(df)
        
        # Additional check
        if len(X) != len(y):
            print(f"⚠️  Final data mismatch: X({len(X)}) != y({len(y)})")
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            if X_scaled is not None:
                X_scaled = X_scaled[:min_len]
            print(f"✅ Fixed: Now both have {min_len} samples")
        
        print(f"✅ Prepared {len(X)} samples with {X.shape[1]-1} features (+ intercept)")
        return X, y, X_scaled
    except Exception as e:
        print(f"❌ Failed to prepare features: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def compare_models(ols_results, ann_results=None):
    """Compare OLS and ANN model performance"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON: OLS vs ANN")
    print("="*60)
    
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ols_performance': ols_results,
    }
    
    if ann_results:
        comparison['ann_performance'] = ann_results
        
        print("\n   Performance Comparison:")
        print("   " + "-"*40)
        print("   Metric        | OLS         | ANN         | Difference")
        print("   " + "-"*40)
        
        metrics = ['mae', 'rmse', 'r2']
        for metric in metrics:
            if metric in ols_results and metric in ann_results:
                ols_val = ols_results[metric]
                ann_val = ann_results[metric]
                
                if metric == 'r2':
                    diff = ann_val - ols_val
                    better = "ANN" if diff > 0 else "OLS"
                else:
                    diff = ols_val - ann_val
                    better = "ANN" if diff > 0 else "OLS"
                
                diff_pct = (abs(diff) / ols_val * 100) if ols_val != 0 else 0
                
                if metric == 'r2':
                    print(f"   {metric.upper():12} | {ols_val:.4f}      | {ann_val:.4f}      | {diff:+.4f} ({diff_pct:+.1f}%) - Better: {better}")
                else:
                    print(f"   {metric.upper():12} | ${ols_val:.2f}     | ${ann_val:.2f}     | ${diff:+.2f} ({diff_pct:+.1f}%) - Better: {better}")
        
        print("\n   📈 Summary:")
        if ann_results.get('r2', 0) > ols_results.get('r2', 0):
            print("     ANN has better explanatory power (higher R²)")
        else:
            print("     OLS has better explanatory power (higher R²)")
            
        if ann_results.get('mae', 0) < ols_results.get('mae', 0):
            print("     ANN has better prediction accuracy (lower MAE)")
        else:
            print("     OLS has better prediction accuracy (lower MAE)")
    
    # Save comparison results
    os.makedirs('learningBase/comparison', exist_ok=True)
    comp_path = 'learningBase/comparison/model_comparison.json'
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n   💾 Comparison saved to {comp_path}")
    return comparison

def main():
    """Main training pipeline for OLS model"""
    print("="*70)
    print("📈 ORDINARY LEAST SQUARES (OLS) PRICE PREDICTOR TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    # Step 1: Load data
    print("\n1️⃣ LOADING DATA")
    print("-"*50)
    df = create_sample_data()
    
    if df is None or len(df) == 0:
        print("❌ No data available for training")
        return None, None
    
    print(f"   Samples: {len(df)}")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"   Average price: ${df['price'].mean():.2f}")
    
    # Step 2: Initialize OLS predictor
    print("\n2️⃣ INITIALIZING OLS PREDICTOR")
    print("-"*50)
    predictor = OLSPricePredictor()
    print("✅ OLSPricePredictor initialized")
    
    # Step 3: Prepare data
    print("\n3️⃣ PREPARING DATA FOR OLS")
    print("-"*50)
    X, y, X_scaled = prepare_data_for_ols(df, predictor)
    
    if X is None or y is None:
        print("❌ Data preparation failed")
        return None, None
    
    # Step 4: Split data
    print("\n4️⃣ SPLITTING DATA")
    print("-"*50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if X_scaled is not None:
        X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    else:
        X_train_scaled, X_test_scaled = None, None
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Step 5: Train OLS model
    print("\n5️⃣ TRAINING OLS MODEL")
    print("-"*50)
    ols_result = predictor.train_model(X_train, y_train, X_test, y_test)
    
    if ols_result is None:
        print("❌ OLS training failed")
        return None, None
    
    # Step 6: Model evaluation summary
    print("\n6️⃣ MODEL EVALUATION SUMMARY")
    print("-"*50)
    
    # Display OLS statistics
    if hasattr(ols_result, 'rsquared'):
        print(f"   R-squared: {ols_result.rsquared:.4f}")
        print(f"   Adjusted R-squared: {ols_result.rsquared_adj:.4f}")
        print(f"   F-statistic: {ols_result.fvalue:.2f}")
        print(f"   Prob (F-statistic): {ols_result.f_pvalue:.4e}")
    
    # Step 7: Create visualizations
    print("\n7️⃣ CREATING VISUALIZATIONS")
    print("-"*50)
    predictor.create_visualizations(X_train, y_train, X_test, y_test, X_test_scaled)
    
    # Step 8: Save model
    print("\n8️⃣ SAVING MODEL")
    print("-"*50)
    predictor.save_model()
    
    # Step 9: Test predictions
    print("\n9️⃣ TESTING PREDICTIONS")
    print("-"*50)
    
    # Test with example products
    examples = [
        {
            'condition_encoded': 5,  # New
            'age': 1,
            'title_length': 50,
            'has_defects': 0,
            'shipping_cost': 0,
            'brand_encoded': 0,
            'category_encoded': 0,
            'product_year': 2023
        },
        {
            'condition_encoded': 2,  # Fair
            'age': 6,
            'title_length': 30,
            'has_defects': 1,
            'shipping_cost': 25,
            'brand_encoded': 0,
            'category_encoded': 0,
            'product_year': 2017
        }
    ]
    
    for i, features in enumerate(examples, 1):
        prediction = predictor.predict_price(features)
        if prediction:
            print(f"\n   Example {i}:")
            print(f"     Predicted: ${prediction['predicted_price']:,.2f}")
            print(f"     Range: ${prediction['price_range']['low']:,.2f} - ${prediction['price_range']['high']:,.2f}")
            print(f"     Confidence: {prediction['confidence_score']:.1%}")
            print(f"     Std Error: ${prediction['std_error']}")
            if prediction.get('r_squared'):
                print(f"     R²: {prediction['r_squared']}")
    
    # Step 10: Calculate evaluation metrics
    print("\n🔟 CALCULATING EVALUATION METRICS")
    print("-"*50)
    
    eval_results = None
    if hasattr(predictor, 'predictions') and predictor.predictions is not None:
        # Calculate metrics from predictions
        if len(predictor.predictions) > 0 and len(y_test) > 0:
            min_len = min(len(predictor.predictions), len(y_test))
            y_test_trimmed = y_test[:min_len]
            pred_trimmed = predictor.predictions[:min_len]
            
            try:
                mae = mean_absolute_error(y_test_trimmed, pred_trimmed)
                rmse = np.sqrt(mean_squared_error(y_test_trimmed, pred_trimmed))
                r2 = r2_score(y_test_trimmed, pred_trimmed)
                
                eval_results = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                print(f"   Test MAE: ${mae:.2f}")
                print(f"   Test RMSE: ${rmse:.2f}")
                print(f"   Test R²: {r2:.4f}")
                
                # Save evaluation results
                eval_path = os.path.join(predictor.learning_base_path, 'test_evaluation.json')
                with open(eval_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"   📊 Test evaluation saved to {eval_path}")
                
            except Exception as e:
                print(f"   ⚠️  Error calculating metrics: {e}")
    else:
        print("   ⚠️  No predictions available for evaluation")
    
    # Step 11: Final summary
    print("\n📋 TRAINING SUMMARY")
    print("-"*50)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Training Time: {total_time:.1f} seconds")
    print(f"📊 OLS Performance:")
    if hasattr(ols_result, 'rsquared'):
        print(f"   R-squared: {ols_result.rsquared:.4f}")
        print(f"   Adjusted R-squared: {ols_result.rsquared_adj:.4f}")
    if eval_results:
        print(f"   Test MAE: ${eval_results['mae']:.2f}")
        print(f"   Test RMSE: ${eval_results['rmse']:.2f}")
        print(f"   Test R²: {eval_results['r2']:.4f}")
    
    print(f"\n💾 Files Created:")
    print(f"   Model: {predictor.model_path}")
    print(f"   Summary: {predictor.model_summary_path}")
    print(f"   Visualizations: {predictor.learning_base_path}*.png")
    
    print(f"\n📈 Key Advantages of OLS:")
    print("   1. Statistical significance tests for each feature")
    print("   2. Confidence intervals for predictions")
    print("   3. Interpretable coefficients")
    print("   4. Assumption checking (normality, homoscedasticity)")
    
    print("\n" + "="*70)
    print("✅ OLS PRICE PREDICTOR TRAINING COMPLETE!")
    print("="*70)
    
    return predictor, eval_results

def load_ann_results():
    """Load ANN results for comparison"""
    ann_results_path = 'learningBase/training_summary.csv'
    ann_json_path = 'learningBase/training_history.json'
    
    ann_results = None
    
    # Try CSV first
    if os.path.exists(ann_results_path):
        try:
            ann_df = pd.read_csv(ann_results_path)
            if not ann_df.empty:
                ann_results = {
                    'mae': ann_df['final_val_mae'].iloc[0] if 'final_val_mae' in ann_df.columns else 0,
                    'rmse': np.sqrt(ann_df['final_val_mse'].iloc[0]) if 'final_val_mse' in ann_df.columns else 0,
                    'r2': ann_df['final_val_r2'].iloc[0] if 'final_val_r2' in ann_df.columns else 0
                }
                print(f"✅ Loaded ANN results from {ann_results_path}")
        except Exception as e:
            print(f"⚠️  Could not load ANN CSV: {e}")
    
    # Try JSON if CSV failed
    if ann_results is None and os.path.exists(ann_json_path):
        try:
            with open(ann_json_path, 'r') as f:
                ann_data = json.load(f)
                ann_results = {
                    'mae': ann_data.get('final_validation_mae', 0),
                    'rmse': np.sqrt(ann_data.get('final_validation_mse', 0)) if ann_data.get('final_validation_mse') else 0,
                    'r2': ann_data.get('final_validation_r2', 0)
                }
                print(f"✅ Loaded ANN results from {ann_json_path}")
        except Exception as e:
            print(f"⚠️  Could not load ANN JSON: {e}")
    
    return ann_results

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("learningBase/ols", exist_ok=True)
    os.makedirs("learningBase/comparison", exist_ok=True)
    
    print("🔧 Setting up directories...")
    print(f"   Data directory: {'data/'} {'✓' if os.path.exists('data/') else '✗'}")
    print(f"   Models directory: {'models/'} {'✓' if os.path.exists('models/') else '✗'}")
    print(f"   Learning base: {'learningBase/'} {'✓' if os.path.exists('learningBase/') else '✗'}")
    print(f"   OLS learning base: {'learningBase/ols/'} {'✓' if os.path.exists('learningBase/ols/') else '✗'}")
    
    # Run OLS training
    predictor, ols_results = main()
    
    if predictor:
        if ols_results:
            print("\n🎉 OLS Model trained successfully with evaluation results!")
            
            # Ask if user wants to compare with ANN
            compare = input("\nCompare with ANN model? (y/n): ")
            if compare.lower() == 'y':
                ann_results = load_ann_results()
                if ann_results:
                    comparison = compare_models(ols_results, ann_results)
                    
                    # Create comparison visualization
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(10, 6))
                        metrics = ['MAE ($)', 'RMSE ($)', 'R²']
                        ols_values = [ols_results['mae'], ols_results['rmse'], ols_results['r2']]
                        ann_values = [ann_results['mae'], ann_results['rmse'], ann_results['r2']]
                        
                        x = np.arange(len(metrics))
                        width = 0.35
                        
                        plt.bar(x - width/2, ols_values, width, label='OLS', color='blue', alpha=0.7)
                        plt.bar(x + width/2, ann_values, width, label='ANN', color='orange', alpha=0.7)
                        
                        plt.xlabel('Metrics')
                        plt.ylabel('Value')
                        plt.title('Model Comparison: OLS vs ANN')
                        plt.xticks(x, metrics)
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Add value labels
                        for i, v in enumerate(ols_values):
                            plt.text(i - width/2, v + max(max(ols_values), max(ann_values))*0.01, 
                                    f'{v:.2f}' if i < 2 else f'{v:.3f}', 
                                    ha='center', va='bottom')
                        
                        for i, v in enumerate(ann_values):
                            plt.text(i + width/2, v + max(max(ols_values), max(ann_values))*0.01, 
                                    f'{v:.2f}' if i < 2 else f'{v:.3f}', 
                                    ha='center', va='bottom')
                        
                        comp_viz_path = 'learningBase/comparison/model_comparison_chart.png'
                        plt.savefig(comp_viz_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"📊 Comparison chart saved to {comp_viz_path}")
                    except Exception as e:
                        print(f"⚠️  Could not create comparison chart: {e}")
                else:
                    print("⚠️  ANN results not found. Run ANN training first (python main.py).")
                    compare_models(ols_results)
            else:
                print("\n📊 OLS Results:")
                print(f"   MAE: ${ols_results['mae']:.2f}")
                print(f"   RMSE: ${ols_results['rmse']:.2f}")
                print(f"   R²: {ols_results['r2']:.4f}")
        else:
            print("\n⚠️  OLS Model trained but evaluation results incomplete.")
            print("   You can still use the model for predictions.")
        
        print("\n🔮 Using the OLS Model:")
        print("   from ols_price_predictor import OLSPricePredictor")
        print("   predictor = OLSPricePredictor()")
        print("   predictor.load_model()")
        print("   prediction = predictor.predict_price(features_dict)")
    else:
        print("\n❌ OLS training failed.")