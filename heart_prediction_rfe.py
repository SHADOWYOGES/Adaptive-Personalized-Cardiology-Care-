import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heart (1) (1).csv')
print(f"Dataset shape: {df.shape}")
print(f"\nDataset columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nMissing values:")
print(df.isnull().sum())

# Separate features and targets
# For target prediction: exclude 'target' and 'trestbps' from features
# For trestbps prediction: exclude 'target' and 'trestbps' from features
feature_cols = [col for col in df.columns if col not in ['target', 'trestbps']]
print(f"\nFeature columns (excluding target and trestbps): {feature_cols}")

X_all = df[feature_cols]
y_target = df['target']
y_trestbps = df['trestbps']

# Split data for target model
X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
    X_all, y_target, test_size=0.2, random_state=42, stratify=y_target
)

# Split data for trestbps model
X_train_trestbps, X_test_trestbps, y_train_trestbps, y_test_trestbps = train_test_split(
    X_all, y_trestbps, test_size=0.2, random_state=42
)

print("\n" + "="*80)
print("MODEL 1: Predicting TARGET (Heart Disease Classification)")
print("="*80)

# RFE for Target (Classification)
print("\nPerforming RFE for Target prediction...")
# Using RandomForestClassifier as base estimator for RFE
rfe_target = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5)
rfe_target.fit(X_train_target, y_train_target)

selected_features_target = X_train_target.columns[rfe_target.support_].tolist()
print(f"\nSelected features for Target model (top 5): {selected_features_target}")
print(f"\nFeature rankings:")
for i, (feature, rank) in enumerate(zip(X_train_target.columns, rfe_target.ranking_)):
    print(f"  {feature}: Rank {rank} {'✓' if rank == 1 else ''}")

# Train model with selected features for Target
X_train_target_selected = X_train_target[selected_features_target]
X_test_target_selected = X_test_target[selected_features_target]

target_model = RandomForestClassifier(n_estimators=100, random_state=42)
target_model.fit(X_train_target_selected, y_train_target)

y_pred_target = target_model.predict(X_test_target_selected)
y_pred_proba_target = target_model.predict_proba(X_test_target_selected)

# Classify into High/Medium/Low risk based on probability
# High risk: prob > 0.7, Medium: 0.3 <= prob <= 0.7, Low: prob < 0.3
def classify_risk(prob):
    if prob >= 0.7:
        return 'High'
    elif prob >= 0.3:
        return 'Medium'
    else:
        return 'Low'

y_pred_category_target = [classify_risk(prob[1]) for prob in y_pred_proba_target]
accuracy = accuracy_score(y_test_target, y_pred_target)

print(f"\nTarget Model Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_target, y_pred_target))

# Save target model and RFE
os.makedirs('models', exist_ok=True)
joblib.dump(target_model, 'models/target_model.joblib')
joblib.dump(rfe_target, 'models/rfe_target.joblib')
joblib.dump(selected_features_target, 'models/selected_features_target.joblib')
print(f"\n✓ Target model saved to models/target_model.joblib")
print(f"✓ RFE object saved to models/rfe_target.joblib")
print(f"✓ Selected features saved to models/selected_features_target.joblib")

print("\n" + "="*80)
print("MODEL 2: Predicting TRESTBPS (Resting Blood Pressure - Regression)")
print("="*80)

# RFE for Trestbps (Regression)
print("\nPerforming RFE for Trestbps prediction...")
# Using RandomForestRegressor as base estimator for RFE
rfe_trestbps = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=5)
rfe_trestbps.fit(X_train_trestbps, y_train_trestbps)

selected_features_trestbps = X_train_trestbps.columns[rfe_trestbps.support_].tolist()
print(f"\nSelected features for Trestbps model (top 5): {selected_features_trestbps}")
print(f"\nFeature rankings:")
for i, (feature, rank) in enumerate(zip(X_train_trestbps.columns, rfe_trestbps.ranking_)):
    print(f"  {feature}: Rank {rank} {'✓' if rank == 1 else ''}")

# Train model with selected features for Trestbps
X_train_trestbps_selected = X_train_trestbps[selected_features_trestbps]
X_test_trestbps_selected = X_test_trestbps[selected_features_trestbps]

trestbps_model = RandomForestRegressor(n_estimators=100, random_state=42)
trestbps_model.fit(X_train_trestbps_selected, y_train_trestbps)

y_pred_trestbps = trestbps_model.predict(X_test_trestbps_selected)

# Classify trestbps into High/Medium/Low based on blood pressure ranges
# High: >= 140 (Hypertension Stage 1), Medium: 120-139 (Elevated), Low: < 120 (Normal)
def classify_bp(bp_value):
    if bp_value >= 140:
        return 'High'
    elif bp_value >= 120:
        return 'Medium'
    else:
        return 'Low'

y_pred_category_trestbps = [classify_bp(bp) for bp in y_pred_trestbps]
y_actual_category_trestbps = [classify_bp(bp) for bp in y_test_trestbps]

mse = mean_squared_error(y_test_trestbps, y_pred_trestbps)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_trestbps, y_pred_trestbps)
r2 = r2_score(y_test_trestbps, y_pred_trestbps)

print(f"\nTrestbps Model Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Root Mean Squared Error: {rmse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")

# Save trestbps model and RFE
joblib.dump(trestbps_model, 'models/trestbps_model.joblib')
joblib.dump(rfe_trestbps, 'models/rfe_trestbps.joblib')
joblib.dump(selected_features_trestbps, 'models/selected_features_trestbps.joblib')
print(f"\n✓ Trestbps model saved to models/trestbps_model.joblib")
print(f"✓ RFE object saved to models/rfe_trestbps.joblib")
print(f"✓ Selected features saved to models/selected_features_trestbps.joblib")

print("\n" + "="*80)
print("COMMON FEATURES SUMMARY")
print("="*80)
common_features = set(selected_features_target) & set(selected_features_trestbps)
print(f"\nCommon features selected for both models: {list(common_features) if common_features else 'None'}")
print(f"\nTarget model features: {selected_features_target}")
print(f"Trestbps model features: {selected_features_trestbps}")

print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
print("\nTarget Model - Feature Importance:")
target_importance = pd.DataFrame({
    'Feature': selected_features_target,
    'Importance': target_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(target_importance.to_string(index=False))

print("\nTrestbps Model - Feature Importance:")
trestbps_importance = pd.DataFrame({
    'Feature': selected_features_trestbps,
    'Importance': trestbps_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(trestbps_importance.to_string(index=False))

print("\n" + "="*80)
print("PREDICTION EXAMPLES WITH PROBABILITY/CATEGORY")
print("="*80)
print("\nSample predictions for Target (with probability and risk category):")
sample_indices = [0, 1, 2, 3, 4]
for idx in sample_indices:
    if idx < len(X_test_target_selected):
        pred = target_model.predict(X_test_target_selected.iloc[[idx]])[0]
        prob = target_model.predict_proba(X_test_target_selected.iloc[[idx]])[0]
        category = y_pred_category_target[idx]
        actual = y_test_target.iloc[idx]
        print(f"  Sample {idx}: Actual={actual}, Predicted={pred}, Prob=[{prob[0]:.3f}, {prob[1]:.3f}], "
              f"Risk Category={category}, {'✓' if actual == pred else '✗'}")

print("\nSample predictions for Trestbps (with BP category):")
for idx in sample_indices:
    if idx < len(X_test_trestbps_selected):
        pred = trestbps_model.predict(X_test_trestbps_selected.iloc[[idx]])[0]
        pred_category = y_pred_category_trestbps[idx]
        actual = y_test_trestbps.iloc[idx]
        actual_category = y_actual_category_trestbps[idx]
        error = abs(actual - pred)
        print(f"  Sample {idx}: Actual={actual:.1f} ({actual_category}), Predicted={pred:.1f} ({pred_category}), "
              f"Error={error:.1f}, {'✓' if actual_category == pred_category else '✗'}")

print("\n" + "="*80)
print("CATEGORY DISTRIBUTION")
print("="*80)
print("\nTarget Risk Category Distribution (Predicted):")
target_category_counts = pd.Series(y_pred_category_target).value_counts()
print(target_category_counts)

print("\nTrestbps BP Category Distribution (Predicted):")
trestbps_category_counts = pd.Series(y_pred_category_trestbps).value_counts()
print(trestbps_category_counts)
print("\nTrestbps BP Category Distribution (Actual):")
trestbps_actual_counts = pd.Series(y_actual_category_trestbps).value_counts()
print(trestbps_actual_counts)

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)

