# Parkinson's Disease Prediction

## Overview
This project aims to predict Parkinson's disease using machine learning techniques. The dataset used is the UCI Parkinson's Disease dataset (ID: 174), which contains biomedical voice measurements from healthy individuals and Parkinson's patients. The pipeline includes data preprocessing, feature selection, dimensionality reduction, and classification using machine learning models such as Random Forest and Support Vector Machines (SVM).

## Dataset
The dataset consists of 195 instances with 23 attributes, extracted from voice recordings. These attributes include fundamental frequency measurements, variation parameters, and nonlinear dynamic properties.

### Key Features:
- **MDVP:Fo(Hz)** - Fundamental frequency of voice
- **MDVP:Fhi(Hz), MDVP:Flo(Hz)** - Maximum and minimum fundamental frequencies
- **Jitter (local, absolute, RAP, PPQ5)** - Variations in fundamental frequency
- **Shimmer (local, dB, APQ3, APQ5, APQ11, DDA)** - Variations in voice amplitude
- **Harmonic-to-Noise Ratio (HNR)** - Ratio of harmonic components to noise
- **Spread1, Spread2, PPE** - Nonlinear measures representing voice signal instability

### Target Variable:
- **`1`** - Parkinson's Disease positive
- **`0`** - Healthy individual

## Comparison of Approaches

This project includes two different implementations:

1. **Python Script (`parkinson.py`)**: A standalone script that processes the dataset, performs feature selection, trains machine learning models, and evaluates their performance. It is designed for quick execution and automation.
2. **Jupyter Notebook (`dr_parkinson.ipynb`)**: An interactive approach that provides detailed step-by-step execution with data visualizations, feature importance analysis, and intermediate outputs. This is useful for exploration and debugging.

### Key Differences
| Feature | Python Script | Jupyter Notebook |
|---------|--------------|-----------------|
| Execution | Fully automated | Step-by-step execution |
| Visualization | Limited | Extensive EDA and plots |
| Debugging | Harder to debug | Easier to debug with outputs |
| Use Case | Batch processing, automation | Interactive exploration, analysis |

## Workflow

### 1. Data Preprocessing
- **Handling Missing Values**: Ensures no missing values remain
- **Removing Duplicates**: Prevents redundant information from affecting the model
- **Feature Scaling**: Uses `StandardScaler` to normalize all features to a standard range

### 2. Feature Selection
- A Random Forest classifier is used to calculate feature importance scores
- The top 10 most important features are selected for model training

### 3. Exploratory Data Analysis
- **Class Distribution**: Visualizes the proportion of Parkinson’s and non-Parkinson’s cases
- **Box Plots & Violin Plots**: Examine the spread and density of key features
- **Correlation Matrix**: Identifies relationships between features

### 4. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduces data to two principal components
- **Linear Discriminant Analysis (LDA)**: Optimizes class separability while reducing dimensions
- **Truncated Singular Value Decomposition (SVD)**: Performs dimensionality reduction without centering data
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Projects high-dimensional data into 2D for visualization

### 5. Model Training & Evaluation
- **Machine Learning Models Used:**
  - **Random Forest Classifier**
  - **Support Vector Machine (SVM)**
- **Hyperparameter Tuning:**
  - Performed using `GridSearchCV` with `StratifiedKFold` cross-validation
  - Random Forest: `n_estimators` (50, 100, 200), `max_depth` (None, 10, 20)
  - SVM: `C` (0.1, 1, 10), `kernel` (linear, rbf)
- **Performance Metrics:**
  - **Accuracy Score**: Measures overall correctness of predictions
  - **Precision & Recall**: Evaluates class-wise predictive performance
  - **F1-Score**: Balances precision and recall
  - **Confusion Matrix**: Provides an intuitive visual of classification performance

## Expected Outputs
- **Feature Importance Plot**: Displays the top 10 influential features in classification
- **Class Distribution Plot**: Shows the proportion of Parkinson’s vs. non-Parkinson’s cases
- **Dimensionality Reduction Visualizations**: Scatter plots using PCA, LDA, SVD, and t-SNE
- **Model Performance Metrics:**
  - **Expected Accuracy**: 85-95% depending on feature selection
  - **Confusion Matrix**: Highlights classification results
  - **Classification Report**: Provides precision, recall, and F1-score

## Usage
To run the full pipeline, execute the script `parkinson.py`:
```bash
python parkinson.py
```
To interactively explore the analysis, open and run `dr_parkinson.ipynb` in Jupyter Notebook:
```bash
jupyter notebook dr_parkinson.ipynb
```

## Dependencies
Ensure the following Python libraries are installed before running the script:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
```

## Conclusion
This project provides an efficient pipeline for detecting Parkinson's disease using voice features. By leveraging feature selection, dimensionality reduction, and classification models, the approach achieves high accuracy in predicting Parkinson's disease. The combination of Random Forest and SVM ensures robust classification performance.

The Python script is ideal for automated execution, while the Jupyter Notebook allows for in-depth analysis and visualization. Both approaches complement each other, making this a comprehensive solution for Parkinson’s disease prediction.
