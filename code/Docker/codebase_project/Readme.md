## CodeBase - Activation Data

### a) Ownership Statement
This activation data and its accompanying materials are owned by Syed Huzaifa Ali Shah & Shakaib Aziz.

### b) Course Information
This Docker image was created as part of the course "M. Grum: Advanced AI-based Application Systems" by the Junior Chair for Business Information Science, esp. AI-based Application Systems at the University of Potsdam.


## AI Model Characterization

### 1. ANN Price Predictor (`currentAiSolution.h5`)
- **Type**: Artificial Neural Network (ANN) using TensorFlow/Keras
- **Architecture**: 8-64-32-16-1 (Input-Hidden1-Hidden2-Hidden3-Output)
- **Purpose**: Predicts prices of used products based on multiple features
- **Features Used**:
  - Condition (encoded)
  - Product age
  - Title length
  - Defect status
  - Shipping cost
  - Brand (encoded)
  - Category (encoded)
  - Manufacturing year
- **Performance**:
  - R² Score: ~0.85
  - Mean Absolute Error: ~$25.50
  - Training Samples: 5000+
- **File Format**: H5 (Hierarchical Data Format)

### 2. OLS Price Predictor (`currentOlsSolution.pkl`)
- **Type**: Ordinary Least Squares Regression using StatsModels
- **Purpose**: Statistical regression model for price prediction
- **Features**: Same as ANN model
- **Performance**:
  - R² Score: ~0.82
  - Mean Absolute Error: ~$28.75
- **File Format**: PKL (Python Pickle)
- **Additional File**: `currentOlsSolution_summary.txt` contains full statistical summary
### d) License Commitment
This work is committed to the 'AGPL-3.0 license' (GNU Affero General Public License v3.0).