Adaptive Personalized Cardiology Care
Adaptive Personalized Cardiology Care is an AI-driven system for real-time cardiac health prediction and personalized recommendations. It integrates multimodal data—wearable ECG signals, clinical parameters, and lifestyle metrics—into dual machine learning models that predict Cardiovascular Disease (CVD) risk and Blood Pressure (BP).
The project includes a full-featured Streamlit dashboard with analytics, visualizations, and an AI recommendation engine.

✅ Features
1. CVD Risk Prediction Model
Predicts:

CVD probability (%)
Risk category (Low → Critical)
Factor-wise risk contribution
Inputs: age, chest pain type, max heart rate, ST depression, thallium test.

2. BP Prediction Model
Predicts:
Systolic & diastolic BP
BP category (Normal / Elevated / Hypertension)
Inputs: age, heart rate, BMI, sodium intake.

3. Streamlit Dashboard
Includes:
Real-time assessment forms
Automatically generated parameters
Trend graphs (Plotly)
Risk category distribution
Factor analysis
Separate tabs for CVD, BP, Analytics, and AI Recommendations

4. AI Recommendation Engine
Uses OpenRouter API to generate:
Diet plans
Lifestyle guidance
Warning signs
Personalized health insights

5. SQLite Health Database

Stores:
CVD assessments
BP assessments
Timestamps
Risk scores
Used for historical trend analysis.

Project Structure
├── dual_model_health_app.py          # Main Streamlit dashboard
├── heart_prediction_rfe.py           # ML training script (CVD)
├── heart_prediction_rfe_advanced.py  # ML training (advanced RFE)
├── predict_with_loaded_models.py     # Model testing script
├── heart (1) (1).csv                 # Heart dataset
├── health_dashboard.db               # SQLite database
├── med.env                           # Environment variables
├── requirements.txt                  # Dependencies

How to Operate (Step-by-Step)
1. Install Dependencies
Run the following command:
pip install -r requirements.txt

2. Add API Key (Optional for AI Recommendations)
Open the file:
med.env
Add:
OPENROUTER_KEY=your_api_key_here

3. Launch the Dashboard
Run:
streamlit run dual_model_health_app.py

4. Using the Dashboard
Tab 1: CVD Risk Assessment
Enter:
Age
Chest pain type
Allow auto-generated HR & ST depression
Thallium test result
Click "Assess Risk"
View:
Risk %
Risk category
Factor-level analysis
Input summary
Scroll down to see assessment history.

Tab 2: Blood Pressure Prediction
Age & Heart Rate auto-filled (can regenerate)
Enter:
Weight
Height
Sodium intake
Click "Predict Blood Pressure"
View:
Systolic/Diastolic values
BP classification
Health warnings if needed

Tab 3: Analytics & Visualizations
Includes:
Line graph for CVD trends
Pie chart for risk distribution
Bar charts for factor analysis
BP trend visualizations

Tab 4: AI Recommendations
After completing CVD or BP assessment → click:
"Get Fresh Recommendations"
The system generates:
Diet plans
Lifestyle tips
Warning signs
Long-term heart care advice

5. Stored Database
The system automatically saves:
Timestamp
Input features
Model output
Category labels
Database file:
health_dashboard.db

Dataset
Uses heart-disease dataset containing:
ECG-related values
Chest pain type
Stress test results
Vitals (BP, max heart rate)
Target label

 Disclaimer
This system is for research and educational purposes only.
Not intended for real medical diagnosis or treatment.
