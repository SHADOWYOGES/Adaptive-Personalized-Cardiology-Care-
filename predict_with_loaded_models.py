"""
Example script to load saved models and make predictions
This demonstrates how to use the saved joblib models
"""
import pandas as pd
import numpy as np
import joblib

# Load the saved models and feature selectors
print("Loading saved models...")
target_model = joblib.load('models/target_model.joblib')
trestbps_model = joblib.load('models/trestbps_model.joblib')
rfe_target = joblib.load('models/rfe_target.joblib')
rfe_trestbps = joblib.load('models/rfe_trestbps.joblib')
selected_features_target = joblib.load('models/selected_features_target.joblib')
selected_features_trestbps = joblib.load('models/selected_features_trestbps.joblib')

print("✓ Models loaded successfully!\n")

# Load test data
df = pd.read_csv('heart (1) (1).csv')
feature_cols = [col for col in df.columns if col not in ['target', 'trestbps']]
X_all = df[feature_cols]

# Example: Make predictions on first 5 samples
print("="*80)
print("MAKING PREDICTIONS WITH LOADED MODELS")
print("="*80)

sample_data = X_all.iloc[:5]
print("\nSample input data:")
print(sample_data[selected_features_target])

# Prepare features for each model
X_target_selected = sample_data[selected_features_target]
X_trestbps_selected = sample_data[selected_features_trestbps]

# Predict target (heart disease)
target_predictions = target_model.predict(X_target_selected)
target_probabilities = target_model.predict_proba(X_target_selected)

# Classify risk categories
def classify_risk(prob):
    if prob >= 0.7:
        return 'High'
    elif prob >= 0.3:
        return 'Medium'
    else:
        return 'Low'

target_risk_categories = [classify_risk(prob[1]) for prob in target_probabilities]

# Predict trestbps (blood pressure)
trestbps_predictions = trestbps_model.predict(X_trestbps_selected)

# Classify BP categories
def classify_bp(bp_value):
    if bp_value >= 140:
        return 'High'
    elif bp_value >= 120:
        return 'Medium'
    else:
        return 'Low'

trestbps_categories = [classify_bp(bp) for bp in trestbps_predictions]

# Display results
print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

for i in range(len(sample_data)):
    print(f"\nSample {i+1}:")
    print(f"  Target Prediction: {target_predictions[i]} (Probability: [{target_probabilities[i][0]:.3f}, {target_probabilities[i][1]:.3f}])")
    print(f"  Risk Category: {target_risk_categories[i]}")
    print(f"  Trestbps Prediction: {trestbps_predictions[i]:.1f} mmHg")
    print(f"  BP Category: {trestbps_categories[i]}")
    print(f"  Actual Target: {df.iloc[i]['target']}")
    print(f"  Actual Trestbps: {df.iloc[i]['trestbps']:.1f} mmHg")

print("\n" + "="*80)
print("Models are ready for production use!")
print("="*80)

