import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heart (1) (1).csv')
print(f"Dataset shape: {df.shape}\n")

# Separate features and targets
feature_cols = [col for col in df.columns if col not in ['target', 'trestbps']]
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

# Function to find optimal number of features using cross-validation
def find_optimal_features(X_train, y_train, model_type='classification', max_features=None):
    """Find optimal number of features using cross-validation"""
    if max_features is None:
        max_features = min(10, X_train.shape[1])
    
    best_score = -np.inf if model_type == 'regression' else 0
    best_n_features = 1
    scores = []
    
    for n in range(1, max_features + 1):
        if model_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'r2'
        
        rfe = RFE(estimator=estimator, n_features_to_select=n)
        rfe.fit(X_train, y_train)
        
        cv_scores = cross_val_score(estimator, X_train.iloc[:, rfe.support_], y_train, 
                                    cv=5, scoring=scoring)
        mean_score = cv_scores.mean()
        scores.append((n, mean_score))
        
        if model_type == 'regression':
            if mean_score > best_score:
                best_score = mean_score
                best_n_features = n
        else:
            if mean_score > best_score:
                best_score = mean_score
                best_n_features = n
    
    return best_n_features, scores

print("="*80)
print("MODEL 1: Predicting TARGET (Heart Disease Classification)")
print("="*80)

# Find optimal number of features for Target
print("\nFinding optimal number of features for Target model...")
optimal_n_target, cv_scores_target = find_optimal_features(
    X_train_target, y_train_target, model_type='classification'
)
print(f"Optimal number of features: {optimal_n_target}")

# RFE for Target
print(f"\nPerforming RFE for Target prediction (selecting {optimal_n_target} features)...")
rfe_target = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), 
                 n_features_to_select=optimal_n_target)
rfe_target.fit(X_train_target, y_train_target)

selected_features_target = X_train_target.columns[rfe_target.support_].tolist()
print(f"\nSelected features for Target model: {selected_features_target}")

# Train model with selected features
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
joblib.dump(target_model, 'models/target_model_advanced.joblib')
joblib.dump(rfe_target, 'models/rfe_target_advanced.joblib')
joblib.dump(selected_features_target, 'models/selected_features_target_advanced.joblib')
print(f"\n✓ Target model saved to models/target_model_advanced.joblib")
print(f"✓ RFE object saved to models/rfe_target_advanced.joblib")
print(f"✓ Selected features saved to models/selected_features_target_advanced.joblib")

print("\n" + "="*80)
print("MODEL 2: Predicting TRESTBPS (Resting Blood Pressure - Regression)")
print("="*80)

# Find optimal number of features for Trestbps
print("\nFinding optimal number of features for Trestbps model...")
optimal_n_trestbps, cv_scores_trestbps = find_optimal_features(
    X_train_trestbps, y_train_trestbps, model_type='regression'
)
print(f"Optimal number of features: {optimal_n_trestbps}")

# RFE for Trestbps
print(f"\nPerforming RFE for Trestbps prediction (selecting {optimal_n_trestbps} features)...")
rfe_trestbps = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), 
                   n_features_to_select=optimal_n_trestbps)
rfe_trestbps.fit(X_train_trestbps, y_train_trestbps)

selected_features_trestbps = X_train_trestbps.columns[rfe_trestbps.support_].tolist()
print(f"\nSelected features for Trestbps model: {selected_features_trestbps}")

# Train model with selected features
X_train_trestbps_selected = X_train_trestbps[selected_features_trestbps]
X_test_trestbps_selected = X_test_trestbps[selected_features_trestbps]

# Try multiple regression models
models_trestbps = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0)
}

best_model_trestbps = None
best_r2 = -np.inf
best_model_name = None

print("\nTesting multiple regression models...")
for name, model in models_trestbps.items():
    model.fit(X_train_trestbps_selected, y_train_trestbps)
    y_pred = model.predict(X_test_trestbps_selected)
    r2 = r2_score(y_test_trestbps, y_pred)
    print(f"  {name}: R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_trestbps = model
        best_model_name = name

print(f"\nBest model: {best_model_name}")

y_pred_trestbps = best_model_trestbps.predict(X_test_trestbps_selected)

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

print(f"\nTrestbps Model Performance ({best_model_name}):")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Root Mean Squared Error: {rmse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")

# Save trestbps model and RFE
joblib.dump(best_model_trestbps, 'models/trestbps_model_advanced.joblib')
joblib.dump(rfe_trestbps, 'models/rfe_trestbps_advanced.joblib')
joblib.dump(selected_features_trestbps, 'models/selected_features_trestbps_advanced.joblib')
print(f"\n✓ Trestbps model saved to models/trestbps_model_advanced.joblib")
print(f"✓ RFE object saved to models/rfe_trestbps_advanced.joblib")
print(f"✓ Selected features saved to models/selected_features_trestbps_advanced.joblib")

print("\n" + "="*80)
print("COMMON FEATURES SUMMARY")
print("="*80)
common_features = set(selected_features_target) & set(selected_features_trestbps)
print(f"\nCommon features selected for both models: {list(common_features) if common_features else 'None'}")
print(f"\nTarget model features ({len(selected_features_target)}): {selected_features_target}")
print(f"Trestbps model features ({len(selected_features_trestbps)}): {selected_features_trestbps}")

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
if hasattr(best_model_trestbps, 'feature_importances_'):
    trestbps_importance = pd.DataFrame({
        'Feature': selected_features_trestbps,
        'Importance': best_model_trestbps.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    # For Ridge regression, use absolute coefficients
    trestbps_importance = pd.DataFrame({
        'Feature': selected_features_trestbps,
        'Importance': np.abs(best_model_trestbps.coef_)
    }).sort_values('Importance', ascending=False)
print(trestbps_importance.to_string(index=False))

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print("\nTarget Model - CV Scores by number of features:")
for n, score in cv_scores_target:
    marker = " ← Optimal" if n == optimal_n_target else ""
    print(f"  {n} features: {score:.4f}{marker}")

print("\nTrestbps Model - CV Scores by number of features:")
for n, score in cv_scores_trestbps:
    marker = " ← Optimal" if n == optimal_n_trestbps else ""
    print(f"  {n} features: {score:.4f}{marker}")

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
        pred = best_model_trestbps.predict(X_test_trestbps_selected.iloc[[idx]])[0]
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

