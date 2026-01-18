# ols_price_predictor.py - Complete Fixed OLS Regression Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import joblib
import warnings
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
from scipy import stats

warnings.filterwarnings('ignore')

class OLSPricePredictor:
    def __init__(self):
        self.model = None
        self.ols_result = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.ols_summary = None
        self.learning_base_path = 'learningBase/ols/'
        self.model_path = 'currentOlsSolution.pkl'
        self.model_summary_path = 'currentOlsSolution_summary.txt'
        self.residuals = None
        self.predictions = None
        
        # Create learning base directory
        os.makedirs(self.learning_base_path, exist_ok=True)
        
    def prepare_features(self, df):
        """Prepare features for OLS training"""
        print("   Preparing features for OLS...")
        
        # Encode categorical variables
        categorical_cols = ['brand', 'category']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Define feature columns (same as ANN for fair comparison)
        self.feature_columns = [
            'condition_encoded', 'age', 'title_length', 
            'has_defects', 'shipping_cost', 'brand_encoded',
            'category_encoded', 'product_year'
        ]
        
        # Keep only columns that exist
        self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        # Prepare features and target
        X = df[self.feature_columns].fillna(0).astype(np.float64)
        y = df['price'].fillna(df['price'].mean()).astype(np.float64)
        
        # FIX: Ensure X and y have same number of samples
        if len(X) != len(y):
            print(f"   ⚠️  Data mismatch: X has {len(X)} samples, y has {len(y)} samples")
            min_samples = min(len(X), len(y))
            X = X.iloc[:min_samples]
            y = y.iloc[:min_samples]
            print(f"   ✅ Trimmed to {min_samples} samples")
        
        # Scale features (helps with interpretation)
        X_scaled = self.scaler.fit_transform(X)
        
        # Add constant term for OLS (intercept)
        X_scaled_with_const = sm.add_constant(X_scaled)
        
        return X_scaled_with_const, y.values, X_scaled
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train OLS model using Statsmodels - FIXED: Made X_val and y_val optional"""
        print("   Training OLS model with Statsmodels...")
        
        try:
            # Fit OLS model
            self.model = sm.OLS(y_train, X_train)
            self.ols_result = self.model.fit()
            
            # Store summary
            self.ols_summary = self.ols_result.summary()
            
            print(f"\n   ✅ OLS Training Completed!")
            print(f"     R-squared: {self.ols_result.rsquared:.4f}")
            print(f"     Adjusted R-squared: {self.ols_result.rsquared_adj:.4f}")
            print(f"     F-statistic: {self.ols_result.fvalue:.2f}")
            print(f"     Prob (F-statistic): {self.ols_result.f_pvalue:.4e}")
            
            # Display coefficient summary
            print("\n   📊 Top 5 Most Significant Coefficients:")
            params_df = pd.DataFrame({
                'Coefficient': self.ols_result.params,
                'Std Error': self.ols_result.bse,
                't-value': self.ols_result.tvalues,
                'P>|t|': self.ols_result.pvalues
            }).sort_values('P>|t|').head(6)
            
            for idx, row in params_df.iterrows():
                coef_name = 'Intercept' if idx == 'const' else f'Feature {idx}'
                significance = '***' if row['P>|t|'] < 0.001 else '**' if row['P>|t|'] < 0.01 else '*' if row['P>|t|'] < 0.05 else ''
                print(f"     {coef_name}: {row['Coefficient']:8.4f} (p={row['P>|t|']:.4e}) {significance}")
            
            # If validation data provided, evaluate
            if X_val is not None and y_val is not None:
                self.evaluate_model(X_val, y_val)
            
            return self.ols_result
            
        except Exception as e:
            print(f"   ❌ OLS training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, X_test, y_test, X_test_scaled=None):
        """Evaluate OLS model performance"""
        print("\n   📈 Evaluating OLS model...")
        
        if not self.ols_result:
            print("   ❌ No trained model available")
            return None
        
        # Make predictions
        y_pred = self.ols_result.predict(X_test)
        self.predictions = y_pred
        self.residuals = y_test - y_pred
        
        # FIX: Ensure arrays have same length
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]
        self.residuals = self.residuals[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Adjusted R-squared for test set
        n = len(y_test)
        p = X_test.shape[1] - 1  # Excluding constant
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        print(f"     Test MAE: ${mae:.2f}")
        print(f"     Test RMSE: ${rmse:.2f}")
        print(f"     Test R²: {r2:.4f}")
        print(f"     Test Adjusted R²: {adj_r2:.4f}")
        
        # Store evaluation results
        eval_results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'adj_r2': adj_r2,
            'n_test_samples': n,
            'n_features': p,
            'timestamp': datetime.now().strftime('%Y-%m-d %H:%M:%S')
        }
        
        # Save evaluation results
        eval_path = os.path.join(self.learning_base_path, 'ols_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"   📊 Evaluation saved to {eval_path}")
        
        # Additional diagnostics
        self._calculate_diagnostics(y_test, y_pred, X_test_scaled)
        
        return eval_results
    
    def _calculate_diagnostics(self, y_true, y_pred, X_scaled):
        """Calculate OLS diagnostic statistics"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        # Durbin-Watson test for autocorrelation
        diff_residuals = np.diff(residuals)
        dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        # Breusch-Pagan test for heteroscedasticity (simplified)
        residuals_sq = residuals**2
        if X_scaled is not None and len(X_scaled) > 0:
            X_with_const = sm.add_constant(X_scaled[:len(residuals_sq)])
            if len(residuals_sq) == X_with_const.shape[0]:
                bp_model = sm.OLS(residuals_sq, X_with_const).fit()
                bp_stat = bp_model.rsquared * n
                bp_pvalue = 1 - stats.chi2.cdf(bp_stat, X_scaled.shape[1])
            else:
                bp_stat, bp_pvalue = None, None
        else:
            bp_stat, bp_pvalue = None, None
        
        diagnostics = {
            'durbin_watson': dw,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_pvalue': bp_pvalue,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Save diagnostics
        diag_path = os.path.join(self.learning_base_path, 'ols_diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
        print(f"\n   🔍 OLS Diagnostics:")
        print(f"     Durbin-Watson: {dw:.3f} (≈2 indicates no autocorrelation)")
        print(f"     Jarque-Bera p-value: {jb_pvalue:.4f} (<0.05 suggests non-normal residuals)")
        if bp_stat:
            print(f"     Breusch-Pagan p-value: {bp_pvalue:.4f} (<0.05 suggests heteroscedasticity)")
        print(f"     Residual Skewness: {diagnostics['skewness']:.3f} (0=symmetric)")
        print(f"     Residual Kurtosis: {diagnostics['kurtosis']:.3f} (3=normal)")
    
    def create_visualizations(self, X_train, y_train, X_test, y_test, X_test_scaled=None):
        """Create diagnostic and scatter plots for OLS"""
        print("\n   🎨 Creating OLS visualizations...")
        
        # Make predictions if not already done
        if self.predictions is None:
            self.predictions = self.ols_result.predict(X_test)
        if self.residuals is None:
            self.residuals = y_test - self.predictions
        
        # FIX: Ensure arrays have same length
        min_len = min(len(y_test), len(self.predictions))
        y_test = y_test[:min_len]
        self.predictions = self.predictions[:min_len]
        self.residuals = self.residuals[:min_len]
        
        # Create figure for all visualizations
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Residuals vs Fitted Plot
        ax1 = plt.subplot(3, 4, 1)
        if len(self.predictions) > 0 and len(self.residuals) > 0:
            ax1.scatter(self.predictions, self.residuals, alpha=0.6)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Fitted Values (Predicted Prices)')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Fitted Values')
        else:
            ax1.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # 2. QQ Plot for normality
        ax2 = plt.subplot(3, 4, 2)
        if len(self.residuals) > 0:
            stats.probplot(self.residuals, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normality Check)')
        else:
            ax2.text(0.5, 0.5, 'No residuals data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scale-Location Plot (homoscedasticity)
        ax3 = plt.subplot(3, 4, 3)
        if len(self.predictions) > 0 and len(self.residuals) > 0:
            sqrt_abs_resid = np.sqrt(np.abs(self.residuals))
            ax3.scatter(self.predictions, sqrt_abs_resid, alpha=0.6)
            ax3.set_xlabel('Fitted Values')
            ax3.set_ylabel('√|Standardized Residuals|')
            ax3.set_title('Scale-Location Plot (Homoscedasticity)')
        else:
            ax3.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Scale-Location Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals Histogram
        ax4 = plt.subplot(3, 4, 4)
        if len(self.residuals) > 0:
            ax4.hist(self.residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
            
            # Add normal distribution curve
            mu, std = np.mean(self.residuals), np.std(self.residuals)
            xmin, xmax = ax4.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax4.plot(x, p, 'k', linewidth=2, label=f'N({mu:.1f}, {std:.1f}²)')
            
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Density')
            ax4.set_title('Residuals Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No residuals data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Residuals Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Actual vs Predicted Scatter (Test)
        ax5 = plt.subplot(3, 4, 5)
        if len(y_test) > 0 and len(self.predictions) > 0:
            ax5.scatter(y_test, self.predictions, alpha=0.6, color='orange')
            min_val = min(min(y_test), min(self.predictions))
            max_val = max(max(y_test), max(self.predictions))
            ax5.plot([min_val, max_val], [min_val, max_val], 
                    'r--', label='Perfect Prediction')
            ax5.set_xlabel('Actual Prices (Test)')
            ax5.set_ylabel('Predicted Prices (Test)')
            ax5.set_title('Test Set: Actual vs Predicted')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No test data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Actual vs Predicted')
        ax5.grid(True, alpha=0.3)
        
        # 6. Prediction Error Plot
        ax6 = plt.subplot(3, 4, 6)
        if len(self.predictions) > 0 and len(self.residuals) > 0:
            ax6.scatter(self.predictions, self.residuals, alpha=0.6)
            ax6.axhline(y=0, color='r', linestyle='--')
            ax6.set_xlabel('Predicted Prices')
            ax6.set_ylabel('Prediction Error')
            ax6.set_title('Prediction Errors')
        else:
            ax6.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Prediction Errors')
        ax6.grid(True, alpha=0.3)
        
        # 7. Coefficient Plot
        ax7 = plt.subplot(3, 4, 7)
        if hasattr(self.ols_result, 'params'):
            coefs = self.ols_result.params[1:]  # Exclude intercept
            if len(coefs) > 0:
                # Get top 10 coefficients by absolute value
                n_coefs = min(10, len(coefs))
                top_indices = np.argsort(np.abs(coefs))[-n_coefs:]
                top_coefs = coefs[top_indices]
                
                y_pos = np.arange(n_coefs)
                ax7.barh(y_pos, top_coefs)
                ax7.set_yticks(y_pos)
                ax7.set_yticklabels([f'Feature {i+1}' for i in top_indices])
                ax7.set_xlabel('Coefficient Value')
                ax7.set_title(f'Top {n_coefs} Feature Coefficients')
            else:
                ax7.text(0.5, 0.5, 'No coefficients', ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Coefficients')
        else:
            ax7.text(0.5, 0.5, 'No model parameters', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Coefficients')
        ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Cook's Distance (influence plot)
        ax8 = plt.subplot(3, 4, 8)
        if hasattr(self.ols_result, 'get_influence'):
            try:
                influence = self.ols_result.get_influence()
                cooks_d = influence.cooks_distance[0]
                if len(cooks_d) > 0:
                    ax8.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
                    ax8.axhline(y=4/len(cooks_d), color='r', linestyle='--', 
                               label='4/n threshold')
                    ax8.set_xlabel('Observation Index')
                    ax8.set_ylabel("Cook's Distance")
                    ax8.set_title("Cook's Distance (Influence)")
                    ax8.legend()
                else:
                    ax8.text(0.5, 0.5, 'No influence data', ha='center', va='center', transform=ax8.transAxes)
                    ax8.set_title("Cook's Distance")
            except:
                ax8.text(0.5, 0.5, 'Error calculating influence', ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title("Cook's Distance")
        else:
            ax8.text(0.5, 0.5, 'No influence data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title("Cook's Distance")
        ax8.grid(True, alpha=0.3)
        
        # 9. Leverage vs Residuals
        ax9 = plt.subplot(3, 4, 9)
        if hasattr(self.ols_result, 'get_influence') and len(self.residuals) > 0:
            try:
                influence = self.ols_result.get_influence()
                leverage = influence.hat_matrix_diag
                if len(leverage) == len(self.residuals):
                    ax9.scatter(leverage, self.residuals, alpha=0.6)
                    ax9.axhline(y=0, color='r', linestyle='--')
                    ax9.set_xlabel('Leverage')
                    ax9.set_ylabel('Residuals')
                    ax9.set_title('Leverage vs Residuals')
                else:
                    ax9.text(0.5, 0.5, 'Data size mismatch', ha='center', va='center', transform=ax9.transAxes)
                    ax9.set_title('Leverage vs Residuals')
            except:
                ax9.text(0.5, 0.5, 'Error calculating leverage', ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Leverage vs Residuals')
        else:
            ax9.text(0.5, 0.5, 'No leverage data', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Leverage vs Residuals')
        ax9.grid(True, alpha=0.3)
        
        # 10. Residuals Autocorrelation
        ax10 = plt.subplot(3, 4, 10)
        if len(self.residuals) > 10:
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(self.residuals, ax=ax10, lags=min(20, len(self.residuals)-1))
                ax10.set_title('Residuals Autocorrelation')
            except:
                ax10.text(0.5, 0.5, 'Error in ACF plot', ha='center', va='center', transform=ax10.transAxes)
                ax10.set_title('Autocorrelation')
        else:
            ax10.text(0.5, 0.5, 'Insufficient data\nfor ACF', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('Autocorrelation')
        ax10.grid(True, alpha=0.3)
        
        # 11. Partial Regression Plots placeholder
        ax11 = plt.subplot(3, 4, 11)
        ax11.text(0.5, 0.5, 'Partial Regression Plots\n(Requires more features)', 
                 ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Partial Regression')
        ax11.grid(True, alpha=0.3)
        
        # 12. Model Performance Summary
        ax12 = plt.subplot(3, 4, 12)
        if hasattr(self.ols_result, 'rsquared'):
            metrics_text = f"""
            Model Performance:
            
            R-squared: {self.ols_result.rsquared:.4f}
            Adj. R²: {self.ols_result.rsquared_adj:.4f}
            F-statistic: {self.ols_result.fvalue:.2f}
            Prob(F): {self.ols_result.f_pvalue:.4e}
            
            Coefficients: {len(self.ols_result.params)}
            Observations: {self.ols_result.nobs}
            """
            ax11.text(0.5, 0.5, metrics_text, ha='center', va='center', transform=ax11.transAxes, fontsize=9)
        else:
            ax11.text(0.5, 0.5, 'No performance metrics', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Model Summary')
        ax11.axis('off')
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        vis_path = os.path.join(self.learning_base_path, 'ols_comprehensive_visualizations.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots for report
        self._create_individual_plots(y_test, self.predictions, self.residuals)
        
        print(f"   📊 Visualizations saved to {self.learning_base_path}")
    
    def _create_individual_plots(self, y_test, predictions, residuals):
        """Create individual plot files for report"""
        
        # 1. Residuals vs Fitted
        if len(predictions) > 0 and len(residuals) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(predictions, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel('Predicted Prices')
            plt.ylabel('Residuals')
            plt.title('OLS Diagnostic Plot: Residuals vs Fitted Values')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.learning_base_path, 'ols_residuals_vs_fitted.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Q-Q Plot
        if len(residuals) > 0:
            plt.figure(figsize=(10, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('OLS Diagnostic Plot: Q-Q Plot for Normality')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.learning_base_path, 'ols_qq_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Actual vs Predicted Scatter
        if len(y_test) > 0 and len(predictions) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.6, color='orange')
            
            min_val = min(min(y_test), min(predictions))
            max_val = max(max(y_test), max(predictions))
            
            plt.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.title('OLS Performance: Actual vs Predicted Prices')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add R² if available
            if hasattr(self.ols_result, 'rsquared'):
                plt.text(0.05, 0.95, f'R² = {self.ols_result.rsquared:.3f}', 
                        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.savefig(os.path.join(self.learning_base_path, 'ols_scatter_plot.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Residuals Distribution
        if len(residuals) > 0:
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(residuals, bins=30, alpha=0.7, 
                                       density=True, edgecolor='black')
            
            # Add normal distribution
            mu, std = np.mean(residuals), np.std(residuals)
            x = np.linspace(min(residuals), max(residuals), 100)
            p = stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Normal Distribution\nμ={mu:.1f}, σ={std:.1f}')
            
            # Add Jarque-Bera test result
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            normality = "Normal" if jb_pvalue > 0.05 else "Not Normal"
            plt.text(0.05, 0.85, f'Jarque-Bera: p={jb_pvalue:.3f}\n({normality})', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.xlabel('Residuals')
            plt.ylabel('Density')
            plt.title('OLS Residuals Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.learning_base_path, 'ols_residuals_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def predict_price(self, product_features):
        """Predict price for a product using OLS"""
        try:
            features_df = pd.DataFrame([product_features])
            
            # Add missing features
            for feature in self.feature_columns:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Reorder columns
            features_df = features_df[self.feature_columns]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df.values)
            
            # Add constant term
            features_with_const = sm.add_constant(features_scaled)
            
            # Make prediction
            if self.ols_result:
                prediction = self.ols_result.predict(features_with_const)[0]
                
                # Calculate prediction interval (95% confidence)
                try:
                    predictions = self.ols_result.get_prediction(features_with_const)
                    pred_frame = predictions.summary_frame(alpha=0.05)
                    
                    low = pred_frame['obs_ci_lower'].iloc[0]
                    high = pred_frame['obs_ci_upper'].iloc[0]
                    std_err = pred_frame['mean_se'].iloc[0]
                except:
                    # Fallback if prediction intervals fail
                    std_err = np.std(self.residuals) if self.residuals is not None else prediction * 0.2
                    low = prediction - 1.96 * std_err
                    high = prediction + 1.96 * std_err
                
                return {
                    'predicted_price': round(float(prediction), 2),
                    'confidence_score': 0.95,
                    'price_range': {
                        'low': round(max(0, low), 2),
                        'high': round(high, 2)
                    },
                    'model_type': 'OLS Regression',
                    'std_error': round(float(std_err), 2),
                    'r_squared': round(float(self.ols_result.rsquared), 4) if hasattr(self.ols_result, 'rsquared') else None
                }
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return None
    
    def save_model(self, path='models/'):
        """Save trained OLS model and components"""
        os.makedirs(path, exist_ok=True)
        
        # Save OLS model data
        model_data = {
            'params': self.ols_result.params if hasattr(self, 'ols_result') and self.ols_result is not None else None,
            'bse': self.ols_result.bse if hasattr(self, 'ols_result') and self.ols_result is not None else None,
            'rsquared': self.ols_result.rsquared if hasattr(self, 'ols_result') and self.ols_result is not None else None,
            'rsquared_adj': self.ols_result.rsquared_adj if hasattr(self, 'ols_result') and self.ols_result is not None else None,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'model_type': 'OLS'
        }
        
        # Save model data
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save model summary as text
        if self.ols_summary:
            with open(self.model_summary_path, 'w') as f:
                f.write(str(self.ols_summary))
        
        # Save other components
        joblib.dump(self.scaler, f'{path}/ols_scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/ols_label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{path}/ols_feature_columns.pkl')
        
        print(f"   Models saved to {path}")
        print(f"   OLS model saved as {self.model_path}")
        print(f"   OLS summary saved as {self.model_summary_path}")
    
    def load_model(self, path='models/'):
        """Load trained model components"""
        try:
            # Load model data
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore attributes
            self.feature_columns = model_data['feature_columns']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            
            print("   OLS model loaded successfully")
            print(f"   Model type: {model_data.get('model_type', 'OLS')}")
            print(f"   R-squared: {model_data.get('rsquared', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Could not load OLS model: {e}")
            return False

# Test function
def test_ols_predictor():
    """Test the OLS predictor"""
    print("Testing OLSPricePredictor class...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'condition': np.random.randint(1, 6, n_samples),
        'age': np.random.randint(0, 10, n_samples),
        'title': ['Product ' + str(i) for i in range(n_samples)],
        'has_defects': np.random.randint(0, 2, n_samples),
        'shipping': np.random.uniform(0, 50, n_samples),
        'brand': ['Brand' + str(i%5) for i in range(n_samples)],
        'category': ['Cat' + str(i%3) for i in range(n_samples)],
        'year': np.random.randint(2015, 2024, n_samples),
        'price': 100 + 20*data['condition'] - 15*data['age'] - 30*data['has_defects'] + np.random.normal(0, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['title_length'] = df['title'].apply(len)
    
    # Initialize and train
    predictor = OLSPricePredictor()
    
    # Prepare features
    X, y, X_scaled = predictor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    ols_result = predictor.train_model(X_train, y_train, X_test, y_test)
    
    if ols_result:
        # Create visualizations
        predictor.create_visualizations(X_train, y_train, X_test, y_test, X_test_scaled)
        
        # Save model
        predictor.save_model()
        
        # Test prediction
        print("\n   Testing OLS prediction...")
        test_features = {
            'condition_encoded': 4,
            'age': 2,
            'title_length': 45,
            'has_defects': 0,
            'shipping_cost': 15.0,
            'brand_encoded': 0,
            'category_encoded': 0,
            'product_year': 2022
        }
        
        prediction = predictor.predict_price(test_features)
        if prediction:
            print(f"     Predicted Price: ${prediction['predicted_price']}")
            print(f"     Price Range: ${prediction['price_range']['low']} - ${prediction['price_range']['high']}")
            print(f"     Confidence: {prediction['confidence_score']:.1%}")
            print(f"     Std Error: ${prediction['std_error']}")
            if prediction['r_squared']:
                print(f"     R-squared: {prediction['r_squared']}")
    
    print("\n✅ OLSPricePredictor test complete!")
    return predictor

if __name__ == "__main__":
    predictor = test_ols_predictor()