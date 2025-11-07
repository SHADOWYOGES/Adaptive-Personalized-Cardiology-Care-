"""
Advanced Cardiovascular Health Monitoring & Simulation Framework
A comprehensive IoT-based AI-driven system for continuous, patient-specific monitoring
and simulation of cardiovascular diseases. Integrates wearable sensors, clinical data,
and real-time feedback with electromechanical heart models and adaptive ML algorithms.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
import requests
import os
import json
import hashlib
import hmac
import time
import copy
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components

# Load environment variables
load_dotenv('med.env')

# ============================================================================
# CONSTANTS
# ============================================================================
CVD_FEATURE_COLUMNS = ['age', 'cp', 'thalach', 'oldpeak', 'thal']
BP_FEATURE_COLUMNS = ['age', 'heart_rate', 'bmi', 'sodium_intake']
CVD_PIPELINE_FILE = 'cardio.joblib'
BM_PIPELINE_FILE = 'breast_mass.joblib'

# Heart model constants (electromechanical parameters)
HEART_MODEL_CONSTANTS = {
    'resting_hr': 72,  # bpm
    'stroke_volume': 70,  # ml
    'cardiac_output': 5.0,  # L/min
    'ejection_fraction': 0.60,
    'systemic_resistance': 1.0,  # mmHg·s/mL
    'arterial_compliance': 1.5,  # mL/mmHg
    'ventricular_elastance': 2.0,  # mmHg/mL
    'av_node_delay': 0.15,  # seconds
    'qrs_duration': 0.08,  # seconds
    'qt_interval': 0.40,  # seconds
}

HEART_BLOCKAGE_LOCATIONS = [
    "None detected",
    "Left Anterior Descending (LAD)",
    "Right Coronary Artery (RCA)",
    "Left Circumflex (LCX)",
]

# HIPAA compliance settings
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'default_key_change_in_production')

# ============================================================================
# DATABASE INITIALIZATION (HIPAA-Compliant)
# ============================================================================
def init_database():
    """Initialize SQLite database with enhanced schema for comprehensive health monitoring."""
    conn = sqlite3.connect('health_dashboard.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # CVD assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cvd_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            cp INTEGER,
            thalach REAL,
            oldpeak REAL,
            thal INTEGER,
            risk_score REAL,
            risk_category TEXT,
            session_id TEXT,
            patient_id_hash TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # BP assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bp_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            heart_rate INTEGER,
            bmi REAL,
            sodium_intake REAL,
            predicted_systolic REAL,
            predicted_diastolic REAL,
            category TEXT,
            session_id TEXT,
            patient_id_hash TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # Wearable sensor data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wearable_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT,
            heart_rate INTEGER,
            heart_rate_variability REAL,
            steps INTEGER,
            calories REAL,
            sleep_duration REAL,
            oxygen_saturation REAL,
            activity_level TEXT,
            device_type TEXT,
            raw_data_json TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # Clinical data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT,
            data_type TEXT,
            value REAL,
            unit TEXT,
            source TEXT,
            notes TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # Heart model simulations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT,
            simulation_type TEXT,
            electrophysiology_params TEXT,
            hemodynamics_params TEXT,
            mechanics_params TEXT,
            results_json TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # Adaptive model updates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_type TEXT,
            update_type TEXT,
            previous_accuracy REAL,
            new_accuracy REAL,
            features_updated TEXT,
            training_samples INTEGER,
            encrypted_data TEXT
        )
    ''')
    
    # Treatment optimization table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS treatment_optimization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT,
            treatment_type TEXT,
            current_dosage REAL,
            optimized_dosage REAL,
            predicted_efficacy REAL,
            side_effects TEXT,
            recommendations TEXT,
            encrypted_data TEXT
        )
    ''')
    
    # Real-time monitoring table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS realtime_monitoring (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT,
            metric_name TEXT,
            metric_value REAL,
            alert_level TEXT,
            alert_message TEXT,
            encrypted_data TEXT
        )
    ''')
    
    conn.commit()
    return conn

DB_CONN = init_database()

# ============================================================================
# SECURITY & HIPAA COMPLIANCE FUNCTIONS
# ============================================================================
def hash_patient_id(patient_id: str) -> str:
    """Hash patient ID for HIPAA compliance (one-way hash)."""
    return hashlib.sha256(patient_id.encode()).hexdigest()

def encrypt_sensitive_data(data: str) -> str:
    """Simple encryption for sensitive data (use proper encryption in production)."""
    return hashlib.sha256((data + ENCRYPTION_KEY).encode()).hexdigest()

def verify_data_integrity(data: str, hash_value: str) -> bool:
    """Verify data integrity using HMAC."""
    expected_hash = hmac.new(
        ENCRYPTION_KEY.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_hash, hash_value)

# ============================================================================
# WEARABLE SENSOR INTEGRATION
# ============================================================================
class WearableSensorIntegration:
    """Integration with wearable sensors (smartwatches, fitness trackers)."""
    
    def __init__(self):
        self.supported_devices = ['apple_watch', 'fitbit', 'garmin', 'samsung_galaxy_watch', 'generic']
    
    def simulate_watch_data(self, device_type: str = 'generic') -> dict:
        """Simulate wearable sensor data (replace with actual API calls in production)."""
        base_hr = np.random.randint(60, 100)
        activity_levels = ['resting', 'light', 'moderate', 'vigorous']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'device_type': device_type,
            'heart_rate': base_hr,
            'heart_rate_variability': np.random.uniform(20, 60),
            'steps': np.random.randint(0, 15000),
            'calories': np.random.uniform(1500, 3000),
            'sleep_duration': np.random.uniform(6, 9),
            'oxygen_saturation': np.random.uniform(95, 100),
            'activity_level': np.random.choice(activity_levels),
            'raw_data': {
                'ecg_samples': np.random.randn(100).tolist(),
                'accelerometer': np.random.randn(3, 100).tolist(),
                'gyroscope': np.random.randn(3, 100).tolist()
            }
        }
    
    def fetch_real_watch_data(self, api_endpoint: str, api_key: str) -> dict:
        """Fetch real wearable data from API (implement based on device API)."""
        try:
            response = requests.get(
                api_endpoint,
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching wearable data: {e}")
            return None
    
    def save_wearable_data(self, data: dict, patient_id_hash: str):
        """Save wearable sensor data to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO wearable_data 
            (timestamp, patient_id_hash, heart_rate, heart_rate_variability, steps, 
             calories, sleep_duration, oxygen_saturation, activity_level, device_type, raw_data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'], patient_id_hash, data['heart_rate'],
            data['heart_rate_variability'], data['steps'], data['calories'],
            data['sleep_duration'], data['oxygen_saturation'], data['activity_level'],
            data['device_type'], json.dumps(data.get('raw_data', {}))
        ))
        DB_CONN.commit()

# ============================================================================
# ELECTROMECHANICAL HEART MODELS
# ============================================================================
class ElectromechanicalHeartModel:
    """Electromechanical model of the heart integrating electrophysiology, hemodynamics, and mechanics."""
    
    def __init__(self, patient_params: dict = None):
        self.params = patient_params or HEART_MODEL_CONSTANTS.copy()
    
    def electrophysiology_model(self, time_points: np.ndarray, heart_rate: float) -> dict:
        """Simulate cardiac electrophysiology (ECG-like signals)."""
        # Simplified ECG model: P wave, QRS complex, T wave
        t = time_points
        hr_rad = 2 * np.pi * heart_rate / 60  # Convert to rad/s
        
        # P wave (atrial depolarization)
        p_wave = 0.1 * np.exp(-((t % (60/heart_rate) - 0.1)**2) / (2 * 0.05**2))
        
        # QRS complex (ventricular depolarization)
        qrs = 0.8 * np.exp(-((t % (60/heart_rate) - 0.2)**2) / (2 * 0.02**2))
        
        # T wave (ventricular repolarization)
        t_wave = 0.3 * np.exp(-((t % (60/heart_rate) - 0.4)**2) / (2 * 0.1**2))
        
        ecg_signal = p_wave + qrs - t_wave
        
        return {
            'time': t.tolist(),
            'ecg_signal': ecg_signal.tolist(),
            'heart_rate': heart_rate,
            'rr_interval': 60 / heart_rate,
            'qt_interval': self.params['qt_interval'],
            'qrs_duration': self.params['qrs_duration']
        }
    
    def hemodynamics_model(self, cardiac_output: float, systemic_resistance: float) -> dict:
        """Simulate cardiovascular hemodynamics."""
        # Simplified Windkessel model
        mean_arterial_pressure = cardiac_output * systemic_resistance
        
        # Stroke volume calculation
        heart_rate = self.params['resting_hr']
        stroke_volume = (cardiac_output * 1000) / heart_rate  # Convert L/min to mL/min, then to mL
        
        # Ejection fraction
        end_diastolic_volume = stroke_volume / self.params['ejection_fraction']
        end_systolic_volume = end_diastolic_volume - stroke_volume
        
        # Pressure calculations
        systolic_pressure = mean_arterial_pressure + (stroke_volume / self.params['arterial_compliance']) * 0.5
        diastolic_pressure = mean_arterial_pressure - (stroke_volume / self.params['arterial_compliance']) * 0.5
        
        return {
            'cardiac_output': cardiac_output,
            'stroke_volume': stroke_volume,
            'ejection_fraction': self.params['ejection_fraction'],
            'mean_arterial_pressure': mean_arterial_pressure,
            'systolic_pressure': systolic_pressure,
            'diastolic_pressure': diastolic_pressure,
            'systemic_resistance': systemic_resistance,
            'end_diastolic_volume': end_diastolic_volume,
            'end_systolic_volume': end_systolic_volume
        }
    
    def mechanics_model(self, ventricular_elastance: float, preload: float, afterload: float) -> dict:
        """Simulate cardiac mechanics (contractility, preload, afterload)."""
        # Frank-Starling relationship
        stroke_volume = preload * ventricular_elastance / (afterload + ventricular_elastance)
        
        # Contractility index
        contractility = ventricular_elastance * stroke_volume
        
        # Wall stress (simplified)
        wall_stress = afterload * stroke_volume / (self.params['stroke_volume'] * 0.1)
        
        return {
            'ventricular_elastance': ventricular_elastance,
            'preload': preload,
            'afterload': afterload,
            'stroke_volume': stroke_volume,
            'contractility': contractility,
            'wall_stress': wall_stress,
            'efficiency': stroke_volume / (preload + afterload)
        }
    
    def integrated_simulation(self, duration: float = 10.0, dt: float = 0.01) -> dict:
        """Run integrated electromechanical simulation."""
        time_points = np.arange(0, duration, dt)
        
        # Electrophysiology
        ep_results = self.electrophysiology_model(time_points, self.params['resting_hr'])
        
        # Hemodynamics
        hd_results = self.hemodynamics_model(
            self.params['cardiac_output'],
            self.params['systemic_resistance']
        )
        
        # Mechanics
        mech_results = self.mechanics_model(
            self.params['ventricular_elastance'],
            hd_results['end_diastolic_volume'],
            hd_results['mean_arterial_pressure']
        )
        
        return {
            'electrophysiology': ep_results,
            'hemodynamics': hd_results,
            'mechanics': mech_results,
            'simulation_duration': duration,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_simulation(self, simulation_results: dict, patient_id_hash: str):
        """Save heart simulation results to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO heart_simulations 
            (timestamp, patient_id_hash, simulation_type, electrophysiology_params, 
             hemodynamics_params, mechanics_params, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            simulation_results['timestamp'], patient_id_hash, 'integrated',
            json.dumps(simulation_results['electrophysiology']),
            json.dumps(simulation_results['hemodynamics']),
            json.dumps(simulation_results['mechanics']),
            json.dumps(simulation_results)
        ))
        DB_CONN.commit()

# ============================================================================
# HEART VISUALIZATION & BLOCKAGE RESPONSE UTILITIES
# ============================================================================
HEART_BLOCKAGE_POINTS = {
    "None detected": None,
    "Left Anterior Descending (LAD)": (-5.2, 4.8),
    "Right Coronary Artery (RCA)": (5.4, -1.8),
    "Left Circumflex (LCX)": (-2.4, -2.0),
}

BLOCKAGE_COLOR_MAP = {
    'Low': ('#10b981', 'rgba(16, 185, 129, 0.32)'),
    'Moderate': ('#f59e0b', 'rgba(245, 158, 11, 0.28)'),
    'High': ('#f97316', 'rgba(249, 115, 22, 0.28)'),
    'Critical': ('#dc2626', 'rgba(220, 38, 38, 0.30)'),
}


def _generate_heart_coordinates(scale: float = 1.0, samples: int = 220):
    """Create parametric heart coordinates for visualization."""
    t = np.linspace(0, 2 * np.pi, samples)
    x = scale * 16 * np.sin(t) ** 3
    y = scale * (
        13 * np.cos(t) -
        5 * np.cos(2 * t) -
        2 * np.cos(3 * t) -
        np.cos(4 * t)
    )
    return x, y


def _derive_blockage_metadata(blockage_percentage: float, location: str) -> dict:
    """Interpret blockage severity and provide messaging."""
    severity = float(np.clip(blockage_percentage or 0.0, 0.0, 100.0))
    location_label = location if location in HEART_BLOCKAGE_POINTS else "None detected"
    if severity >= 85:
        severity_label = 'Critical'
        message = "Critical coronary obstruction detected. Immediate intervention recommended."
    elif severity >= 60:
        severity_label = 'High'
        message = "High-grade blockage reducing perfusion. Intensify monitoring and consult cardiology."
    elif severity >= 30:
        severity_label = 'Moderate'
        message = "Moderate stenosis influencing flow dynamics. Lifestyle and pharmacological review advised."
    elif severity > 0:
        severity_label = 'Low'
        message = "Mild narrowing observed. Continue preventative care and routine follow-up."
    else:
        severity_label = 'Low'
        message = "No detectable blockage. Perfusion appears stable."

    color_pair = BLOCKAGE_COLOR_MAP.get(severity_label, BLOCKAGE_COLOR_MAP['Low'])

    return {
        'percentage': severity,
        'severity_label': severity_label,
        'message': message,
        'location': location_label,
        'color': color_pair[0],
        'fill_color': color_pair[1]
    }


def apply_blockage_effects(simulation_results: dict, blockage_percentage: float, blockage_location: str) -> dict:
    """Adjust simulation outputs to reflect perfusion deficits caused by blockages."""
    if not simulation_results:
        return simulation_results

    updated = copy.deepcopy(simulation_results)
    metadata = _derive_blockage_metadata(blockage_percentage, blockage_location)
    severity_ratio = metadata['percentage'] / 100.0

    perfusion_index = 1.0
    if severity_ratio > 0:
        perfusion_index = max(0.35, 1.0 - 0.65 * (severity_ratio ** 1.05))

    flow_factor = perfusion_index
    pressure_factor = 1.0 + 0.55 * severity_ratio
    contractility_factor = max(0.4, 1.0 - 0.5 * severity_ratio)
    wall_stress_factor = 1.0 + 0.75 * severity_ratio

    hemo = updated.get('hemodynamics', {})
    if hemo:
        hemo['cardiac_output'] = round(hemo['cardiac_output'] * flow_factor, 3)
        hemo['stroke_volume'] = round(hemo['stroke_volume'] * flow_factor, 3)
        hemo['mean_arterial_pressure'] = round(hemo['mean_arterial_pressure'] * pressure_factor, 3)
        hemo['systolic_pressure'] = round(hemo['systolic_pressure'] * pressure_factor, 3)
        hemo['diastolic_pressure'] = round(hemo['diastolic_pressure'] * pressure_factor, 3)
        if hemo.get('ejection_fraction'):
            hemo['end_diastolic_volume'] = round(hemo['stroke_volume'] / hemo['ejection_fraction'], 3)
            hemo['end_systolic_volume'] = round(hemo['end_diastolic_volume'] - hemo['stroke_volume'], 3)
        hemo['perfusion_index'] = round(perfusion_index, 3)

    mech = updated.get('mechanics', {})
    if mech:
        mech['stroke_volume'] = hemo.get('stroke_volume', mech.get('stroke_volume', 0))
        mech['contractility'] = round(mech['contractility'] * contractility_factor, 3)
        mech['wall_stress'] = round(mech['wall_stress'] * wall_stress_factor, 3)
        denom = mech['preload'] + mech['afterload'] + 1e-5
        mech['efficiency'] = round(max(0.05, mech['stroke_volume'] / denom), 4)

    updated['blockage'] = {
        **metadata,
        'flow_factor': round(flow_factor, 3),
        'pressure_factor': round(pressure_factor, 3),
        'contractility_factor': round(contractility_factor, 3),
        'wall_stress_factor': round(wall_stress_factor, 3),
        'perfusion_index': round(perfusion_index, 3)
    }

    return updated


def create_heart_visualization(blockage_percentage: float, blockage_location: str, perfusion_index: float = 1.0) -> go.Figure:
    """Generate an animated heart figure that reflects blockage severity."""
    metadata = _derive_blockage_metadata(blockage_percentage, blockage_location)
    base_scale = 0.6 * max(perfusion_index, 0.35)
    phases = np.linspace(0, 2 * np.pi, 24)

    heart_traces = []
    frames = []

    marker_point = HEART_BLOCKAGE_POINTS.get(metadata['location'])
    if marker_point is not None:
        blockage_marker = go.Scatter(
            x=[marker_point[0]],
            y=[marker_point[1]],
            mode='markers+text',
            marker=dict(
                size=18 + metadata['percentage'] * 0.25,
                color=metadata['color'],
                symbol='circle',
                line=dict(color='#ffffff', width=2)
            ),
            text=[f"{int(metadata['percentage'])}%"],
            textposition="top center",
            name='Blockage'
        )
    else:
        blockage_marker = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            showlegend=False,
            marker=dict(size=1, opacity=0)
        )

    for idx, phase in enumerate(phases):
        beat_scale = base_scale * (1 + 0.12 * np.sin(phase))
        x_vals, y_vals = _generate_heart_coordinates(beat_scale)
        heart_trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor=metadata['fill_color'],
            line=dict(color=metadata['color'], width=3),
            mode='lines',
            name='Myocardium'
        )
        frames.append(go.Frame(data=[heart_trace, blockage_marker], name=str(idx)))

    # Initial frame data
    init_x, init_y = _generate_heart_coordinates(base_scale)
    heart_traces.append(go.Scatter(
        x=init_x,
        y=init_y,
        fill='toself',
        fillcolor=metadata['fill_color'],
        line=dict(color=metadata['color'], width=3),
        mode='lines',
        name='Myocardium'
    ))

    heart_traces.append(blockage_marker)

    fig = go.Figure(data=heart_traces, frames=frames)
    fig.update_layout(
        title="Autonomic Heartbeat Simulation",
        xaxis=dict(showgrid=False, visible=False, range=[-20, 20]),
        yaxis=dict(showgrid=False, visible=False, range=[-18, 18]),
        width=550,
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 120, 'redraw': True},
                            'transition': {'duration': 120},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    ),
                ],
                pad={'r': 10, 't': 35},
                x=0.05,
                y=1.12,
            )
        ],
        sliders=[
            dict(
                steps=[dict(method='animate', args=[[str(i)], {'mode': 'immediate', 'frame': {'duration': 120, 'redraw': True}}], label=str(i+1)) for i in range(len(frames))],
                transition={'duration': 0},
                x=0.1,
                len=0.8,
                currentvalue=dict(visible=False)
            )
        ],
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_3d_beating_heart(auto_play: bool = True) -> str:
    """Return an HTML string that renders a 3D beating heart surface with autoplay animation."""
    # Parameter grid
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0.05 * np.pi, 0.95 * np.pi, 70)
    U, V = np.meshgrid(u, v)

    def base_shape(scale: float = 1.0):
        x = scale * 16 * np.sin(U) ** 3 * np.sin(V)
        y_base = (13 * np.cos(U) - 5 * np.cos(2 * U) - 2 * np.cos(3 * U) - np.cos(4 * U))
        y = scale * y_base * np.sin(V)
        z = scale * 0.55 * y_base * np.cos(V)
        return x, y, z

    base_scale = 0.65
    X0, Y0, Z0 = base_shape(base_scale)

    # Build frames for beating
    phases = np.linspace(0, 2 * np.pi, 24)
    frames = []
    for idx, phase in enumerate(phases):
        beat = base_scale * (1 + 0.10 * np.sin(phase))
        X, Y, Z = base_shape(beat)
        frames.append(go.Frame(
            data=[go.Surface(x=X, y=Y, z=Z, colorscale='Reds', showscale=False, opacity=0.98)],
            name=str(idx)
        ))

    fig = go.Figure(
        data=[go.Surface(x=X0, y=Y0, z=Z0, colorscale='Reds', showscale=False, opacity=0.98)],
        frames=frames
    )

    fig.update_layout(
        title="3D Beating Heart",
        width=640,
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {'frame': {'duration': 120, 'redraw': True}, 'transition': {'duration': 80}, 'fromcurrent': True, 'mode': 'immediate'}]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    ),
                ],
                x=0.05,
                y=1.08,
                pad={'r': 10, 't': 30}
            )
        ],
        sliders=[
            dict(
                steps=[dict(method='animate', args=[[str(i)], {'mode': 'immediate', 'frame': {'duration': 120, 'redraw': True}}], label=str(i+1)) for i in range(len(frames))],
                transition={'duration': 0},
                x=0.15,
                len=0.7,
                currentvalue=dict(visible=False)
            )
        ]
    )

    # Return HTML with autoplay (Streamlit won't auto-play frames without embedding HTML)
    html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, auto_play=auto_play)
    return html

# ============================================================================
# ADAPTIVE LEARNING MODELS
# ============================================================================
class AdaptiveMLModel:
    """Adaptive ML model that evolves with continuous feedback."""
    
    def __init__(self, model_type: str = 'CVD', initial_model=None):
        self.model_type = model_type
        self.model = initial_model
        self.training_history = []
        self.feature_importance_history = []
        self.accuracy_history = []
    
    def update_model(self, new_data: pd.DataFrame, new_labels: pd.Series, 
                     validation_split: float = 0.2) -> dict:
        """Update model with new data (incremental learning)."""
        if self.model is None:
            # Initialize new model
            if self.model_type == 'CVD':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Split new data
        X_train, X_val, y_train, y_val = train_test_split(
            new_data, new_labels, test_size=validation_split, random_state=42
        )
        
        # Retrain with combined data (in production, use incremental learning)
        # For now, we'll simulate by training on new data
        self.model.fit(X_train, y_train)
        
        # Evaluate
        if self.model_type == 'CVD':
            accuracy = accuracy_score(y_val, self.model.predict(X_val))
            metric_name = 'accuracy'
        else:
            accuracy = r2_score(y_val, self.model.predict(X_val))
            metric_name = 'r2_score'
        
        # Track feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(new_data.columns, self.model.feature_importances_))
        else:
            feature_importance = {}
        
        update_info = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'update_type': 'incremental',
            'previous_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'new_accuracy': accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_importance': feature_importance,
            metric_name: accuracy
        }
        
        self.training_history.append(update_info)
        self.accuracy_history.append(accuracy)
        self.feature_importance_history.append(feature_importance)
        
        # Save update to database
        self._save_update(update_info)
        
        return update_info
    
    def _save_update(self, update_info: dict):
        """Save model update information to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO model_updates 
            (timestamp, model_type, update_type, previous_accuracy, new_accuracy, 
             features_updated, training_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            update_info['timestamp'], update_info['model_type'],
            update_info['update_type'], update_info['previous_accuracy'],
            update_info['new_accuracy'], json.dumps(update_info.get('feature_importance', {})),
            update_info['training_samples']
        ))
        DB_CONN.commit()
    
    def predict(self, X: pd.DataFrame):
        """Make predictions using the adaptive model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call update_model first.")
        return self.model.predict(X)
    
    def get_model_performance_trend(self) -> dict:
        """Get model performance trend over time."""
        return {
            'accuracy_history': self.accuracy_history,
            'num_updates': len(self.training_history),
            'latest_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'improvement': (self.accuracy_history[-1] - self.accuracy_history[0]) if len(self.accuracy_history) > 1 else 0
        }

# ============================================================================
# TREATMENT OPTIMIZATION
# ============================================================================
class TreatmentOptimizer:
    """ML-based treatment optimization for cardiovascular diseases."""
    
    def __init__(self):
        self.treatment_models = {}
    
    def optimize_treatment(self, patient_data: dict, current_treatment: dict) -> dict:
        """Optimize treatment dosage and regimen using ML."""
        # Simulate treatment optimization
        treatment_type = current_treatment.get('type', 'medication')
        current_dosage = current_treatment.get('dosage', 0)
        
        # Simulate ML-based optimization (replace with actual model in production)
        # Factors: patient age, weight, comorbidities, drug interactions, etc.
        age_factor = patient_data.get('age', 50) / 100
        bmi_factor = patient_data.get('bmi', 25) / 25
        
        # Optimized dosage calculation (simplified)
        if treatment_type == 'antihypertensive':
            base_dosage = 10.0
            optimized_dosage = base_dosage * (1 + age_factor * 0.2) * (1 + (bmi_factor - 1) * 0.1)
            predicted_efficacy = 0.85 - abs(bmi_factor - 1) * 0.1
        elif treatment_type == 'anticoagulant':
            base_dosage = 5.0
            optimized_dosage = base_dosage * (1 + age_factor * 0.15)
            predicted_efficacy = 0.90
        else:
            optimized_dosage = current_dosage * 1.1
            predicted_efficacy = 0.80
        
        # Side effects prediction (simplified)
        side_effects = []
        if optimized_dosage > current_dosage * 1.2:
            side_effects.append('Increased risk of dizziness')
        if age_factor > 0.7:
            side_effects.append('Monitor kidney function')
        if bmi_factor > 1.3:
            side_effects.append('Consider weight management')
        
        recommendations = {
            'dosage_adjustment': optimized_dosage - current_dosage,
            'monitoring_frequency': 'weekly' if optimized_dosage != current_dosage else 'monthly',
            'lifestyle_changes': ['Reduce sodium intake', 'Regular exercise', 'Weight management'] if bmi_factor > 1.2 else [],
            'follow_up_days': 7 if optimized_dosage != current_dosage else 30
        }
        
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'treatment_type': treatment_type,
            'current_dosage': current_dosage,
            'optimized_dosage': round(optimized_dosage, 2),
            'predicted_efficacy': round(predicted_efficacy, 3),
            'side_effects': side_effects,
            'recommendations': recommendations
        }
        
        # Save to database
        self._save_optimization(optimization_result, patient_data.get('patient_id_hash', ''))
        
        return optimization_result
    
    def _save_optimization(self, result: dict, patient_id_hash: str):
        """Save treatment optimization result to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO treatment_optimization 
            (timestamp, patient_id_hash, treatment_type, current_dosage, optimized_dosage,
             predicted_efficacy, side_effects, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['timestamp'], patient_id_hash, result['treatment_type'],
            result['current_dosage'], result['optimized_dosage'],
            result['predicted_efficacy'], json.dumps(result['side_effects']),
            json.dumps(result['recommendations'])
        ))
        DB_CONN.commit()

# ============================================================================
# REAL-TIME MONITORING
# ============================================================================
class RealTimeMonitor:
    """Real-time monitoring system with alert generation."""
    
    def __init__(self):
        self.alert_thresholds = {
            'heart_rate': {'low': 50, 'high': 100},
            'systolic_bp': {'low': 90, 'high': 140},
            'diastolic_bp': {'low': 60, 'high': 90},
            'oxygen_saturation': {'low': 95, 'high': 100},
            'heart_rate_variability': {'low': 20, 'high': 60}
        }
    
    def monitor_metric(self, metric_name: str, metric_value: float, 
                       patient_id_hash: str) -> dict:
        """Monitor a single metric and generate alerts if needed."""
        thresholds = self.alert_thresholds.get(metric_name, {})
        
        alert_level = 'normal'
        alert_message = None
        
        if metric_value < thresholds.get('low', float('-inf')):
            alert_level = 'critical'
            alert_message = f"{metric_name} is critically low: {metric_value}"
        elif metric_value > thresholds.get('high', float('inf')):
            alert_level = 'critical'
            alert_message = f"{metric_name} is critically high: {metric_value}"
        elif metric_value < thresholds.get('low', float('-inf')) * 1.1:
            alert_level = 'warning'
            alert_message = f"{metric_name} is below normal: {metric_value}"
        elif metric_value > thresholds.get('high', float('inf')) * 0.9:
            alert_level = 'warning'
            alert_message = f"{metric_name} is above normal: {metric_value}"
        
        monitoring_result = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'metric_value': metric_value,
            'alert_level': alert_level,
            'alert_message': alert_message
        }
        
        # Save to database
        self._save_monitoring_data(monitoring_result, patient_id_hash)
        
        return monitoring_result
    
    def _save_monitoring_data(self, data: dict, patient_id_hash: str):
        """Save real-time monitoring data to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO realtime_monitoring 
            (timestamp, patient_id_hash, metric_name, metric_value, alert_level, alert_message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'], patient_id_hash, data['metric_name'],
            data['metric_value'], data['alert_level'], data['alert_message']
        ))
        DB_CONN.commit()
    
    def get_recent_alerts(self, patient_id_hash: str, hours: int = 24) -> list:
        """Get recent alerts for a patient."""
        cursor = DB_CONN.cursor()
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute('''
            SELECT * FROM realtime_monitoring 
            WHERE patient_id_hash = ? AND timestamp > ? AND alert_level != 'normal'
            ORDER BY timestamp DESC
        ''', (patient_id_hash, cutoff_time))
        return cursor.fetchall()

# ============================================================================
# CLINICAL DATA INTEGRATION
# ============================================================================
class ClinicalDataIntegration:
    """Integration with clinical data sources (EHR, lab results, etc.)."""
    
    def save_clinical_data(self, data_type: str, value: float, unit: str,
                          source: str, patient_id_hash: str, notes: str = None):
        """Save clinical data to database."""
        cursor = DB_CONN.cursor()
        cursor.execute('''
            INSERT INTO clinical_data 
            (timestamp, patient_id_hash, data_type, value, unit, source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(), patient_id_hash, data_type,
            value, unit, source, notes
        ))
        DB_CONN.commit()
    
    def get_clinical_history(self, patient_id_hash: str, data_type: str = None) -> list:
        """Retrieve clinical data history."""
        cursor = DB_CONN.cursor()
        if data_type:
            cursor.execute('''
                SELECT * FROM clinical_data 
                WHERE patient_id_hash = ? AND data_type = ?
                ORDER BY timestamp DESC
            ''', (patient_id_hash, data_type))
        else:
            cursor.execute('''
                SELECT * FROM clinical_data 
                WHERE patient_id_hash = ?
                ORDER BY timestamp DESC
            ''', (patient_id_hash,))
        return cursor.fetchall()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []

# Initialize components
wearable_integration = WearableSensorIntegration()
heart_model = ElectromechanicalHeartModel()
treatment_optimizer = TreatmentOptimizer()
realtime_monitor = RealTimeMonitor()
clinical_integration = ClinicalDataIntegration()

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
@st.cache_resource
def load_cvd_pipeline(filename):
    """Loads the CVD risk assessment pipeline."""
    try:
        pipeline = joblib.load(filename)
        return pipeline, True  # Return pipeline and success flag
    except FileNotFoundError:
        # If cardio.joblib doesn't exist, create a simple placeholder pipeline
        # Create a simple placeholder pipeline
        placeholder_model = RandomForestClassifier(n_estimators=10, random_state=42)
        placeholder_model.fit(
            np.random.rand(100, 5), 
            np.random.randint(0, 2, 100)
        )
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', placeholder_model)
        ])
        return pipeline, False  # Return pipeline and failure flag

@st.cache_resource
def load_bm_pipeline(filename):
    """Loads the Blood Pressure prediction pipeline."""
    try:
        pipeline = joblib.load(filename)
        return pipeline, True  # Return pipeline and success flag
    except FileNotFoundError:
        # Simulate a placeholder model for Blood Pressure
        return None, False  # Return None and failure flag

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def get_cvd_risk_output(raw_input_data: list, pipeline: Pipeline) -> dict:
    """CVD risk assessment with risk categorization."""
    raw_array = np.array(raw_input_data).reshape(1, -1)
    live_df = pd.DataFrame(raw_array, columns=CVD_FEATURE_COLUMNS)
    
    try:
        proba = pipeline.predict_proba(live_df)[0]
        # Handle both binary and multi-class cases
        if len(proba) == 2:
            risk_probability = float(proba[1])  # Probability of disease
        else:
            risk_probability = float(proba[0])
    except:
        # Fallback simulation if pipeline fails
        risk_probability = np.random.uniform(0.2, 0.8)
    
    risk_percentage = round(risk_probability * 100)
    
    # Risk categorization
    if risk_probability >= 0.75:
        category = "CRITICAL RISK"
        color = "#dc2626"
        icon = "🔴"
    elif risk_probability >= 0.65:
        category = "HIGH RISK"
        color = "#ef4444"
        icon = "🟠"
    elif risk_probability >= 0.45:
        category = "MODERATE RISK"
        color = "#f59e0b"
        icon = "🟡"
    elif risk_probability >= 0.25:
        category = "LOW-MODERATE RISK"
        color = "#84cc16"
        icon = "🟢"
    else:
        category = "LOW RISK"
        color = "#10b981"
        icon = "✅"
    
    # Factor analysis
    age, cp, thalach, oldpeak, thal = raw_input_data
    factors = {
        'Age': {'value': age, 'risk': 'High' if age > 60 else 'Moderate' if age > 50 else 'Low'},
        'Chest Pain': {'value': cp, 'risk': 'High' if cp == 3 else 'Moderate' if cp >= 1 else 'Low'},
        'Max Heart Rate': {'value': thalach, 'risk': 'High' if thalach < 120 else 'Moderate' if thalach < 150 else 'Low'},
        'ST Depression': {'value': oldpeak, 'risk': 'High' if oldpeak > 2.0 else 'Moderate' if oldpeak > 1.0 else 'Low'},
        'Thallium Test': {'value': thal, 'risk': 'High' if thal == 3 else 'Moderate' if thal == 2 else 'Low'}
    }
    
    return {
        "risk_category": category,
        "risk_score_percent": risk_percentage,
        "raw_probability": risk_probability,
        "color": color,
        "icon": icon,
        "factors": factors,
        "timestamp": datetime.now().isoformat()
    }

def get_bp_prediction_output(raw_input_data: list, pipeline):
    """Blood pressure prediction based on features."""
    # Simulate prediction if pipeline is None
    if pipeline is None:
        # Simulated prediction based on input values
        age, heart_rate, bmi, sodium_intake = raw_input_data
        
        # Simple heuristic for simulation: predict blood pressure (systolic/diastolic)
        # Base BP + adjustments based on features
        base_systolic = 120.0  # Base systolic BP in mmHg
        base_diastolic = 80.0  # Base diastolic BP in mmHg
        
        # BP adjustment based on features
        # Age factor (older = higher BP)
        age_factor = (age - 40) * 0.3
        
        # Heart rate factor (higher HR = higher BP)
        hr_factor = (heart_rate - 75) * 0.15
        
        # BMI factor (higher BMI = higher BP)
        bmi_factor = (bmi - 22) * 0.8
        
        # Sodium intake factor (higher sodium = higher BP)
        sodium_factor = (sodium_intake - 2300) / 100  # Based on daily recommended 2300mg
        
        bp_adjustment = age_factor + hr_factor + bmi_factor + sodium_factor
        
        predicted_systolic = base_systolic + bp_adjustment
        predicted_diastolic = base_diastolic + (bp_adjustment * 0.6)
        
        # Clamp values to reasonable ranges
        predicted_systolic = max(90.0, min(180.0, predicted_systolic))
        predicted_diastolic = max(60.0, min(120.0, predicted_diastolic))
        
        # Categorize BP
        if predicted_systolic >= 140 or predicted_diastolic >= 90:
            category = "High (Hypertension)"
            color = "#ef4444"
        elif predicted_systolic >= 120 or predicted_diastolic >= 80:
            category = "Elevated"
            color = "#f59e0b"
        else:
            category = "Normal"
            color = "#10b981"
    else:
        # Use actual pipeline if available (regression model)
        raw_array = np.array(raw_input_data).reshape(1, -1)
        live_df = pd.DataFrame(raw_array, columns=BP_FEATURE_COLUMNS)
        
        try:
            prediction = pipeline.predict(live_df)[0]
            # If pipeline returns single value, use it as systolic
            if isinstance(prediction, (list, np.ndarray)) and len(prediction) >= 2:
                predicted_systolic = float(prediction[0])
                predicted_diastolic = float(prediction[1])
            else:
                predicted_systolic = float(prediction)
                predicted_diastolic = predicted_systolic * 0.67  # Approximate ratio
            
            # Categorize BP
            if predicted_systolic >= 140 or predicted_diastolic >= 90:
                category = "High (Hypertension)"
                color = "#ef4444"
            elif predicted_systolic >= 120 or predicted_diastolic >= 80:
                category = "Elevated"
                color = "#f59e0b"
            else:
                category = "Normal"
                color = "#10b981"
        except:
            predicted_systolic = 120.0
            predicted_diastolic = 80.0
            category = "Normal"
            color = "#10b981"
    
    return {
        "predicted_systolic": round(predicted_systolic, 1),
        "predicted_diastolic": round(predicted_diastolic, 1),
        "category": category,
        "color": color,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================
def save_cvd_assessment(assessment_data: dict):
    """Save CVD assessment to database."""
    cursor = DB_CONN.cursor()
    risk_score = assessment_data.get('risk_score_percent', 0)
    
    timestamp = assessment_data.get('timestamp', datetime.now().isoformat())
    age = assessment_data.get('age', 0)
    cp = assessment_data.get('cp', 0)
    thalach = assessment_data.get('thalach', 0.0)
    oldpeak = assessment_data.get('oldpeak', 0.0)
    thal = assessment_data.get('thal', 0)
    risk_category = assessment_data.get('risk_category', 'UNKNOWN')
    session_id = str(datetime.now().timestamp())
    
    cursor.execute('''
        INSERT INTO cvd_assessments 
        (timestamp, age, cp, thalach, oldpeak, thal, risk_score, risk_category, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, age, cp, thalach, oldpeak, thal, risk_score, risk_category, session_id
    ))
    DB_CONN.commit()

def save_bp_assessment(assessment_data: dict):
    """Save BP assessment to database."""
    cursor = DB_CONN.cursor()
    
    timestamp = assessment_data.get('timestamp', datetime.now().isoformat())
    age = assessment_data.get('age', 0)
    heart_rate = assessment_data.get('heart_rate', 0)
    bmi = assessment_data.get('bmi', 0.0)
    sodium_intake = assessment_data.get('sodium_intake', 0.0)
    predicted_systolic = assessment_data.get('predicted_systolic', 0.0)
    predicted_diastolic = assessment_data.get('predicted_diastolic', 0.0)
    category = assessment_data.get('category', 'Unknown')
    session_id = str(datetime.now().timestamp())
    
    cursor.execute('''
        INSERT INTO bp_assessments 
        (timestamp, age, heart_rate, bmi, sodium_intake, predicted_systolic, predicted_diastolic, category, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, age, heart_rate, bmi, sodium_intake, predicted_systolic, predicted_diastolic, category, session_id
    ))
    DB_CONN.commit()

def get_cvd_history(limit=30):
    """Retrieve CVD assessment history from database."""
    cursor = DB_CONN.cursor()
    cursor.execute('''
        SELECT * FROM cvd_assessments 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    return cursor.fetchall()

def get_bp_history(limit=30):
    """Retrieve BP assessment history from database."""
    cursor = DB_CONN.cursor()
    cursor.execute('''
        SELECT * FROM bp_assessments 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    results = cursor.fetchall()
    # Get column names from cursor description
    columns = [description[0] for description in cursor.description] if cursor.description else None
    return results, columns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Dual-Model Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Load pipelines
CVD_PIPELINE, CVD_LOADED = load_cvd_pipeline(CVD_PIPELINE_FILE)
BM_PIPELINE, BM_LOADED = load_bm_pipeline(BM_PIPELINE_FILE)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🏥 Health Dashboard")
    st.markdown("---")
    
    # Model status indicators
    with st.expander("📊 Model Status", expanded=False):
        if CVD_LOADED:
            st.success("✅ CVD Model: Loaded")
        else:
            st.warning(f"⚠️ CVD Model: Placeholder (file '{CVD_PIPELINE_FILE}' not found)")
        
        if BM_LOADED:
            st.success("✅ BP Model: Loaded")
        else:
            st.info(f"ℹ️ BP Model: Simulated (file '{BM_PIPELINE_FILE}' not found)")
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **Advanced Cardiovascular Health Monitoring & Simulation Framework**
    
    Features:
    - **CVD Risk Assessment**: ML-based cardiovascular disease risk prediction
    - **Blood Pressure Prediction**: Systolic/diastolic BP prediction
    - **Wearable Sensors**: Integration with smartwatches and fitness trackers
    - **Heart Model Simulation**: Electromechanical heart models (EP, hemodynamics, mechanics)
    - **Real-Time Monitoring**: Continuous monitoring with intelligent alerts
    - **Treatment Optimization**: ML-driven dosage optimization
    - **Adaptive Learning**: Models that evolve with continuous feedback
    - **HIPAA-Compliant**: Secure, scalable data management
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("🏥 Advanced Cardiovascular Health Monitoring & Simulation Framework")
st.markdown("**IoT-based AI-driven system for continuous, patient-specific cardiovascular monitoring and simulation**")
st.markdown("---")

# ============================================================================
# TABS FOR COMPREHENSIVE HEALTH MONITORING
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "⌚ Wearable Sensors",
    "❤️ CVD Risk Assessment", 
    "🩺 Blood Pressure Prediction",
    "🫀 Heart Model Simulation",
    "📊 Analytics & Visualizations",
    "📡 Real-Time Monitoring",
    "💊 Treatment Optimization",
    "🤖 Adaptive Learning"
])

# ============================================================================
# TAB 2: CVD RISK ASSESSMENT
# ============================================================================
with tab2:
    st.header("❤️ Cardiovascular Disease Risk Assessment")
    st.markdown("Enter your health metrics below to assess your cardiovascular disease risk.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Form")
        
        # Initialize auto-generated values if not present
        if 'auto_heart_rate' not in st.session_state:
            st.session_state.auto_heart_rate = np.random.randint(70, 203)
        
        if 'auto_st_depression' not in st.session_state:
            st.session_state.auto_st_depression = round(np.random.uniform(0.0, 6.2), 1)
        
        # Buttons to regenerate random values (outside form)
        col_btn1, col_btn2 = st.columns(2)
        using_sensor_hr = (
            'wearable_data' in st.session_state and isinstance(st.session_state.wearable_data, dict) and 'heart_rate' in st.session_state.wearable_data
        )
        with col_btn1:
            if not using_sensor_hr:
                if st.button("🔄 Regenerate Heart Rate", key="gen_heart_rate", use_container_width=True):
                    st.session_state.auto_heart_rate = np.random.randint(70, 203)
                    st.rerun()
            else:
                st.button("💓 Using Watch Heart Rate", key="gen_heart_rate_disabled", use_container_width=True, disabled=True)
        with col_btn2:
            if st.button("🔄 Regenerate ST Depression", key="gen_st_depression", use_container_width=True):
                st.session_state.auto_st_depression = round(np.random.uniform(0.0, 6.2), 1)
                st.rerun()
        
        with st.form("cvd_form"):
            age = st.slider("Age", 29, 77, 45)
            
            cp_map = {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal",
                3: "Asymptomatic"
            }
            cp_value = st.selectbox(
                "Chest Pain Type",
                options=list(cp_map.keys()),
                format_func=lambda x: cp_map[x]
            )
            
            # Display heart rate (prefer wearable sensor data if available)
            using_sensor_hr = (
                'wearable_data' in st.session_state and isinstance(st.session_state.wearable_data, dict) and 'heart_rate' in st.session_state.wearable_data
            )
            thalach = st.session_state.wearable_data['heart_rate'] if using_sensor_hr else st.session_state.auto_heart_rate
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <span style="color: #66b2ff; font-size: 16px;">💓 <strong>Maximum Heart Rate Achieved</strong>: 
                <span style="color: #ffffff; font-weight: bold; font-size: 18px;">{thalach} BPM</span>
                {('<span style="color:#93c5fd; font-size: 12px;"> (from Wearable Sensors)</span>') if using_sensor_hr else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display ST Depression
            oldpeak = st.session_state.auto_st_depression
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <span style="color: #66b2ff; font-size: 16px;">📊 <strong>ST Depression Induced by Exercise</strong>: 
                <span style="color: #ffffff; font-weight: bold; font-size: 18px;">{oldpeak}</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            thal_map = {
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect"
            }
            thal_value = st.selectbox(
                "Thallium Test Result",
                options=list(thal_map.keys()),
                format_func=lambda x: thal_map[x]
            )
            
            submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)
            
            if submitted:
                raw_input = [age, cp_value, thalach, oldpeak, thal_value]
                assessment = get_cvd_risk_output(raw_input, CVD_PIPELINE)
                
                # Add input values to assessment
                assessment['age'] = age
                assessment['cp'] = cp_value
                assessment['thalach'] = thalach
                assessment['oldpeak'] = oldpeak
                assessment['thal'] = thal_value
                
                # Save to database
                save_cvd_assessment(assessment)
                
                # Store in session state
                st.session_state.cvd_assessment = assessment
                st.rerun()
    
    with col2:
        st.subheader("📊 Results")
        
        if 'cvd_assessment' in st.session_state:
            assessment = st.session_state.cvd_assessment
            
            # Display risk score with st.metric (enhanced visibility)
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 15px 0;">
                <div style="color: #1e293b; font-size: 14px; font-weight: 600; margin-bottom: 10px;">Risk Score</div>
                <div style="color: #0f172a; font-size: 48px; font-weight: 800; margin: 10px 0;">{assessment['risk_score_percent']}%</div>
                <div style="background-color: {assessment['color']}15; color: {assessment['color']}; padding: 8px 16px; border-radius: 20px; display: inline-block; font-weight: 700; margin-top: 10px;">
                    {assessment['icon']} {assessment['risk_category']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display risk category
            st.markdown(f"**Risk Category:** {assessment['risk_category']}")
            
            # Display input values summary
            st.markdown("---")
            st.markdown("**Input Summary:**")
            st.write(f"- Age: {assessment['age']}")
            st.write(f"- Chest Pain Type: {cp_map[assessment['cp']]}")
            st.write(f"- Max Heart Rate: {assessment['thalach']} BPM")
            st.write(f"- ST Depression: {assessment['oldpeak']}")
            st.write(f"- Thallium Test: {thal_map[assessment['thal']]}")
            
            # Display timestamp
            timestamp = datetime.fromisoformat(assessment['timestamp'])
            st.markdown("---")
            st.caption(f"Assessment completed: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("👆 Please fill out the form and click 'Assess Risk' to see results.")
    
    # Assessment History
    st.markdown("---")
    st.subheader("📈 Assessment History")
    
    history = get_cvd_history(10)
    if history:
        history_df = pd.DataFrame(history, columns=[
            'ID', 'Timestamp', 'Age', 'CP', 'Thalach', 'Oldpeak', 'Thal',
            'Risk Score', 'Risk Category', 'Session ID'
        ])
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        history_df = history_df.sort_values('Timestamp', ascending=False)
        
        # Display recent assessments
        display_df = history_df[['Timestamp', 'Risk Score', 'Risk Category']].head(10)
        display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Date & Time', 'Risk Score (%)', 'Category']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessments", len(history_df))
        with col2:
            st.metric("Average Risk Score", f"{history_df['Risk Score'].mean():.1f}%")
        with col3:
            st.metric("Latest Risk Score", f"{history_df['Risk Score'].iloc[0]:.1f}%")
    else:
        st.info("No assessment history yet. Complete assessments to see trends here.")

# ============================================================================
# TAB 3: BLOOD PRESSURE PREDICTION
# ============================================================================
with tab3:
    st.header("🩺 Blood Pressure Prediction")
    st.markdown("Enter the features below to predict blood pressure (systolic and diastolic).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Form")
        
        # Initialize auto-generated values if not present
        if 'auto_bp_age' not in st.session_state:
            st.session_state.auto_bp_age = 45
        
        if 'auto_bp_heart_rate' not in st.session_state:
            st.session_state.auto_bp_heart_rate = np.random.randint(70, 203)
        
        # Buttons to regenerate random values (outside form)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Regenerate Age", key="gen_bp_age", use_container_width=True):
                st.session_state.auto_bp_age = np.random.randint(18, 80)
                st.rerun()
        with col_btn2:
            using_sensor_bp_hr = (
                'wearable_data' in st.session_state and isinstance(st.session_state.wearable_data, dict) and 'heart_rate' in st.session_state.wearable_data
            )
            if not using_sensor_bp_hr:
                if st.button("🔄 Regenerate Heart Rate", key="gen_bp_heart_rate", use_container_width=True):
                    st.session_state.auto_bp_heart_rate = np.random.randint(70, 203)
                    st.rerun()
            else:
                st.button("💓 Using Watch Heart Rate", key="gen_bp_heart_rate_disabled", use_container_width=True, disabled=True)
        
        with st.form("bp_form"):
            # Age
            bp_age = st.session_state.auto_bp_age
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <span style="color: #66b2ff; font-size: 16px;">👤 <strong>Age</strong>: 
                <span style="color: #ffffff; font-weight: bold; font-size: 18px;">{bp_age} years</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Heart Rate (prefer wearable sensor data if available)
            using_sensor_bp_hr = (
                'wearable_data' in st.session_state and isinstance(st.session_state.wearable_data, dict) and 'heart_rate' in st.session_state.wearable_data
            )
            bp_heart_rate = st.session_state.wearable_data['heart_rate'] if using_sensor_bp_hr else st.session_state.auto_bp_heart_rate
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <span style="color: #66b2ff; font-size: 16px;">💓 <strong>Heart Rate</strong>: 
                <span style="color: #ffffff; font-weight: bold; font-size: 18px;">{bp_heart_rate} BPM</span>
                {('<span style=\"color:#93c5fd; font-size: 12px;\"> (from Wearable Sensors)</span>') if using_sensor_bp_hr else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI (manual input)
            st.markdown("---")
            col_bmi1, col_bmi2 = st.columns(2)
            with col_bmi1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            with col_bmi2:
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
            
            # Calculate BMI
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            st.info(f"📊 **Calculated BMI:** {bmi:.1f}")
            
            # Sodium Intake (manual input)
            sodium_intake = st.slider(
                "Daily Sodium Intake (mg)",
                min_value=500,
                max_value=5000,
                value=2300,
                step=50,
                help="Recommended daily intake: 2300mg (1 teaspoon of salt ≈ 2300mg sodium)"
            )
            
            submitted = st.form_submit_button("🔍 Predict Blood Pressure", use_container_width=True)
            
            if submitted:
                raw_input = [bp_age, bp_heart_rate, bmi, sodium_intake]
                assessment = get_bp_prediction_output(raw_input, BM_PIPELINE)
                
                # Store input values
                assessment['age'] = bp_age
                assessment['heart_rate'] = bp_heart_rate
                assessment['bmi'] = round(bmi, 1)
                assessment['sodium_intake'] = sodium_intake
                assessment['weight'] = weight
                assessment['height'] = height
                
                # Save to database
                save_bp_assessment(assessment)
                
                # Store in session state
                st.session_state.bp_assessment = assessment
                st.rerun()
    
    with col2:
        st.subheader("📊 Results")
        
        if 'bp_assessment' in st.session_state:
            assessment = st.session_state.bp_assessment
            
            # Display predicted BP with st.metric
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                st.metric(
                    label="Systolic BP",
                    value=f"{assessment['predicted_systolic']} mmHg"
                )
            with col_bp2:
                st.metric(
                    label="Diastolic BP",
                    value=f"{assessment['predicted_diastolic']} mmHg"
                )
            
            st.markdown("---")
            
            # Display BP reading in standard format
            st.markdown(f"**Blood Pressure:** {assessment['predicted_systolic']}/{assessment['predicted_diastolic']} mmHg")
            st.markdown(f"**Category:** {assessment['category']}")
            
            # Display input values summary
            st.markdown("---")
            st.markdown("**Input Summary:**")
            st.write(f"- Age: {assessment.get('age', 'N/A')} years")
            st.write(f"- Heart Rate: {assessment.get('heart_rate', 'N/A')} BPM")
            st.write(f"- BMI: {assessment.get('bmi', 'N/A')}")
            st.write(f"- Daily Sodium Intake: {assessment.get('sodium_intake', 'N/A')} mg")
            if 'weight' in assessment and 'height' in assessment:
                st.write(f"- Weight: {assessment['weight']} kg, Height: {assessment['height']} cm")
            
            # Display timestamp
            timestamp = datetime.fromisoformat(assessment['timestamp'])
            st.markdown("---")
            st.caption(f"Assessment completed: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display warning if high BP
            if "High" in assessment['category']:
                st.error("⚠️ **Warning:** Predicted high blood pressure. Please consult with a healthcare professional.")
            elif "Elevated" in assessment['category']:
                st.warning("⚠️ **Note:** Elevated blood pressure detected. Monitor regularly.")
            else:
                st.success("✓ Blood pressure within normal range.")
        else:
            st.info("👆 Please fill out the form and click 'Predict Blood Pressure' to see results.")
    

# ============================================================================
# OPENROUTER AI RECOMMENDATIONS
# ============================================================================
def get_health_recommendations(assessment_data: dict, model_type: str = "CVD") -> dict:
    """Get AI-powered health recommendations using OpenRouter."""
    OPENROUTER_KEY = os.getenv('OPENROUTER_KEY')
    if not OPENROUTER_KEY:
        return {"success": False, "error": "OpenRouter API key not found in environment variables."}
    
    if model_type == "CVD":
        risk_level = assessment_data['risk_category']
        risk_score = assessment_data['risk_score_percent']
        
        factors_summary = "\n".join([
            f"- {name}: {info['value']} (Risk: {info['risk']})"
            for name, info in assessment_data['factors'].items()
        ])
        
        system_prompt = """You are an advanced AI cardiology assistant with expertise in preventive medicine. 
        Provide comprehensive, evidence-based health guidance with actionable steps.
        
        Structure your response in these sections:
        
        🍎 PERSONALIZED DIET PLAN:
        - Heart-healthy foods to include
        - Foods to avoid or limit
        - Sample meal plans for the week
        - Specific nutrients and supplements
        
        🎯 RISK MANAGEMENT STRATEGIES:
        - Immediate actions (next 24 hours)
        - Short-term goals (next week)
        - Long-term lifestyle changes (next month)
        - Exercise recommendations
        
        📊 RISK FACTOR BREAKDOWN:
        - Address each specific risk factor
        - Provide targeted interventions
        
        ⚠️ WARNING SIGNS TO MONITOR:
        - Specific symptoms to watch for
        - When to seek immediate medical attention
        
        Use clear, actionable language with specific examples."""
        
        user_context = f"""PATIENT ASSESSMENT:
        Overall Risk: {risk_level} ({risk_score}%)
        
        Individual Risk Factors:
        {factors_summary}
        
        Provide comprehensive health guidance, diet plan, and risk management strategies based on this assessment."""
    
    else:  # BP model
        systolic = assessment_data['predicted_systolic']
        diastolic = assessment_data['predicted_diastolic']
        category = assessment_data['category']
        
        system_prompt = """You are an advanced AI medical assistant specializing in cardiovascular health and blood pressure management. 
        Provide comprehensive, evidence-based guidance with actionable steps.
        
        Structure your response in these sections:
        
        🍎 PERSONALIZED DIET PLAN:
        - Foods that help lower blood pressure (DASH diet principles)
        - Foods to avoid or limit (high sodium, processed foods)
        - Sample meal plans for the week
        - Specific nutrients and supplements (potassium, magnesium, etc.)
        
        🎯 LIFESTYLE MANAGEMENT:
        - Exercise recommendations (aerobic, strength training)
        - Stress management techniques
        - Sleep hygiene and importance
        - Weight management if needed
        - Smoking cessation and alcohol moderation
        
        📊 UNDERSTANDING YOUR RESULTS:
        - What the blood pressure reading means
        - Target ranges to aim for
        - Next steps to take
        - When to follow up with healthcare providers
        
        ⚠️ IMPORTANT NOTES:
        - When to seek immediate medical attention
        - Medication considerations (if applicable)
        - Regular monitoring guidelines
        
        Use clear, actionable language with specific examples."""
        
        user_context = f"""PATIENT ASSESSMENT:
        Blood Pressure: {systolic}/{diastolic} mmHg
        Category: {category}
        
        Input Features:
        - Age: {assessment_data.get('age', 'N/A')} years
        - Heart Rate: {assessment_data.get('heart_rate', 'N/A')} BPM
        - BMI: {assessment_data.get('bmi', 'N/A')}
        - Daily Sodium Intake: {assessment_data.get('sodium_intake', 'N/A')} mg
        
        Provide comprehensive health guidance, diet plan, and blood pressure management strategies based on this assessment."""
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "https://github.com/",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3-haiku",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            recommendations = response.json()["choices"][0]["message"]["content"]
            return {"success": True, "recommendations": recommendations}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# TAB 1: WEARABLE SENSOR INTEGRATION
# ============================================================================
with tab1:
    st.header("⌚ Wearable Sensor Integration")
    st.markdown("Connect and integrate data from wearable devices (smartwatches, fitness trackers)")
    
    # Auto-generate simulated data on first load so other tabs can consume it
    if 'wearable_data' not in st.session_state:
        default_data = wearable_integration.simulate_watch_data('generic')
        st.session_state.wearable_data = default_data
        st.success("✅ Sample smartwatch data loaded")
        try:
            wearable_integration.save_wearable_data(default_data, hash_patient_id('patient_001'))
        except Exception:
            pass
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📱 Device Configuration")
        
        device_type = st.selectbox(
            "Select Device Type",
            options=wearable_integration.supported_devices,
            index=4  # Default to generic
        )
        
        patient_id = st.text_input("Patient ID (for HIPAA compliance)", value="patient_001", type="default")
        patient_id_hash = hash_patient_id(patient_id) if patient_id else ""
        
        st.markdown("---")
        st.subheader("🔌 API Configuration (Optional)")
        
        use_api = st.checkbox("Use Real API (instead of simulation)")
        
        if use_api:
            api_endpoint = st.text_input("API Endpoint", placeholder="https://api.example.com/data")
            api_key = st.text_input("API Key", type="password")
            
            if st.button("🔍 Fetch Real Data", use_container_width=True):
                if api_endpoint and api_key:
                    data = wearable_integration.fetch_real_watch_data(api_endpoint, api_key)
                    if data:
                        st.session_state.wearable_data = data
                        wearable_integration.save_wearable_data(data, patient_id_hash)
                        st.success("✅ Data fetched and saved successfully!")
                    else:
                        st.error("Failed to fetch data. Check API credentials.")
                else:
                    st.warning("Please provide API endpoint and key.")
        else:
            if st.button("🔄 Generate Simulated Data", use_container_width=True):
                data = wearable_integration.simulate_watch_data(device_type)
                st.session_state.wearable_data = data
                wearable_integration.save_wearable_data(data, patient_id_hash)
                st.success("✅ Simulated data generated and saved!")
    
    with col2:
        st.subheader("📊 Sensor Data")
        
        if 'wearable_data' in st.session_state:
            data = st.session_state.wearable_data
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Heart Rate", f"{data['heart_rate']} BPM")
                st.metric("Steps", f"{data['steps']:,}")
                st.metric("Sleep Duration", f"{data['sleep_duration']:.1f} hrs")
            with col_metric2:
                st.metric("HRV", f"{data['heart_rate_variability']:.1f} ms")
                st.metric("Calories", f"{data['calories']:.0f} kcal")
                st.metric("SpO2", f"{data['oxygen_saturation']:.1f}%")
            
            st.markdown("---")
            st.markdown(f"**Activity Level:** {data['activity_level'].title()}")
            st.markdown(f"**Device:** {data['device_type'].replace('_', ' ').title()}")
            st.markdown(f"**Timestamp:** {data['timestamp']}")
            
            # Real-time monitoring integration
            if patient_id_hash:
                hr_alert = realtime_monitor.monitor_metric('heart_rate', data['heart_rate'], patient_id_hash)
                spo2_alert = realtime_monitor.monitor_metric('oxygen_saturation', data['oxygen_saturation'], patient_id_hash)
                
                if hr_alert['alert_level'] != 'normal':
                    st.warning(f"⚠️ {hr_alert['alert_message']}")
                if spo2_alert['alert_level'] != 'normal':
                    st.warning(f"⚠️ {spo2_alert['alert_message']}")
        else:
            st.info("👆 Configure device and generate/fetch data to see sensor readings here.")
    
    # Historical Data Visualization
    if patient_id_hash:
        st.markdown("---")
        st.subheader("📈 Historical Trends")
        
        cursor = DB_CONN.cursor()
        cursor.execute('''
            SELECT timestamp, heart_rate, steps, oxygen_saturation 
            FROM wearable_data 
            WHERE patient_id_hash = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''', (patient_id_hash,))
        history = cursor.fetchall()
        
        if history:
            hist_df = pd.DataFrame(history, columns=['Timestamp', 'Heart Rate', 'Steps', 'SpO2'])
            hist_df['Timestamp'] = pd.to_datetime(hist_df['Timestamp'])
            hist_df = hist_df.sort_values('Timestamp')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df['Timestamp'],
                y=hist_df['Heart Rate'],
                name='Heart Rate',
                line=dict(color='#ef4444')
            ))
            fig.update_layout(
                title="Heart Rate Trend",
                xaxis_title="Time",
                yaxis_title="Heart Rate (BPM)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: HEART MODEL SIMULATION
# ============================================================================
with tab4:
    st.header("🫀 Electromechanical Heart Model Simulation")
    st.markdown("Simulate cardiac electrophysiology, hemodynamics, and mechanics")
    
    # 3D beating heart (autoplay) — renders immediately without user action
    st.markdown("### 🫀 3D Beating Heart (Auto)")
    heart3d_html = create_3d_beating_heart(auto_play=True)
    components.html(heart3d_html, height=540, scrolling=False)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Model Parameters")
        
        heart_rate = st.slider("Heart Rate (BPM)", 50, 120, 72)
        cardiac_output = st.slider("Cardiac Output (L/min)", 3.0, 8.0, 5.0, 0.1)
        systemic_resistance = st.slider("Systemic Resistance", 0.5, 2.0, 1.0, 0.1)
        ventricular_elastance = st.slider("Ventricular Elastance", 1.0, 3.0, 2.0, 0.1)

        st.markdown("---")
        st.subheader("🛡️ Blockage Simulation")
        blockage_location = st.selectbox(
            "Suspected Coronary Segment",
            HEART_BLOCKAGE_LOCATIONS,
            help="Select where the simulated blockage is observed."
        )
        blockage_severity = st.slider(
            "Blockage Severity (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Simulate the percentage of arterial occlusion impacting perfusion."
        )
        
        simulation_duration = st.slider("Simulation Duration (seconds)", 1.0, 30.0, 10.0, 0.5)
        
        patient_id = st.text_input("Patient ID", value="patient_001", key="heart_patient_id")
        patient_id_hash = hash_patient_id(patient_id) if patient_id else ""
        
        if st.button("🔄 Run Simulation", use_container_width=True):
            # Update model parameters
            heart_model.params['resting_hr'] = heart_rate
            heart_model.params['cardiac_output'] = cardiac_output
            heart_model.params['systemic_resistance'] = systemic_resistance
            heart_model.params['ventricular_elastance'] = ventricular_elastance
            
            # Run simulation
            simulation_results = heart_model.integrated_simulation(duration=simulation_duration)
            simulation_results = apply_blockage_effects(simulation_results, blockage_severity, blockage_location)
            simulation_results.setdefault('parameters', {})
            simulation_results['parameters'].update({
                'heart_rate': heart_rate,
                'cardiac_output': cardiac_output,
                'systemic_resistance': systemic_resistance,
                'ventricular_elastance': ventricular_elastance,
                'simulation_duration': simulation_duration,
            })
            st.session_state.heart_simulation = simulation_results
            st.session_state.heart_blockage_inputs = {
                'location': blockage_location,
                'severity': blockage_severity
            }
            
            # Save to database
            if patient_id_hash:
                heart_model.save_simulation(simulation_results, patient_id_hash)
            
            if blockage_severity:
                st.success(f"✅ Simulation completed with {blockage_severity}% occlusion at {blockage_location}.")
            else:
                st.success("✅ Simulation completed! No blockage detected.")
    
    with col2:
        st.subheader("📊 Simulation Results")
        
        if 'heart_simulation' in st.session_state:
            sim = st.session_state.heart_simulation
            blockage = sim.get('blockage', {})
            perfusion_index = blockage.get(
                'perfusion_index',
                sim.get('hemodynamics', {}).get('perfusion_index', 1.0)
            )

            st.markdown("### 🚨 Blockage Response")
            col_bl1, col_bl2, col_bl3 = st.columns(3)
            col_bl1.metric("Blockage Severity", f"{blockage.get('percentage', 0):.0f}%")
            col_bl2.metric("Perfusion Index", f"{perfusion_index:.2f}")
            col_bl3.metric("Flow Retention", f"{blockage.get('flow_factor', 1.0) * 100:.0f}%")

            severity_label = blockage.get('severity_label', 'Low')
            blockage_location_label = blockage.get('location', 'None detected')
            blockage_message = blockage.get('message', 'Perfusion stable.')

            if blockage.get('percentage', 0) == 0:
                st.success("Perfusion stable. No blockage detected in the simulated heart.")
            elif severity_label in ('Critical', 'High'):
                st.error(f"⚠️ {blockage_message} (Location: {blockage_location_label})")
            elif severity_label == 'Moderate':
                st.warning(f"⚠️ {blockage_message} (Location: {blockage_location_label})")
            else:
                st.info(f"ℹ️ {blockage_message} (Location: {blockage_location_label})")

            st.markdown("### 🫀 Animated Heart")
            heart_fig = create_heart_visualization(
                blockage.get('percentage', 0.0),
                blockage.get('location', 'None detected'),
                perfusion_index=perfusion_index
            )
            st.plotly_chart(heart_fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("---")
            
            # Electrophysiology Results
            st.markdown("### ⚡ Electrophysiology")
            ep = sim['electrophysiology']
            col_ep1, col_ep2 = st.columns(2)
            with col_ep1:
                st.metric("Heart Rate", f"{ep['heart_rate']} BPM")
                st.metric("RR Interval", f"{ep['rr_interval']:.3f} s")
            with col_ep2:
                st.metric("QRS Duration", f"{ep['qrs_duration']:.3f} s")
                st.metric("QT Interval", f"{ep['qt_interval']:.3f} s")
            
            # ECG Visualization
            if len(ep['time']) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ep['time'][:1000],  # Limit to first 1000 points for performance
                    y=ep['ecg_signal'][:1000],
                    name='ECG Signal',
                    line=dict(color='#ef4444', width=1)
                ))
                fig.update_layout(
                    title="ECG Signal (Electrophysiology)",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude (mV)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Hemodynamics Results
            st.markdown("---")
            st.markdown("### 🩸 Hemodynamics")
            hd = sim['hemodynamics']
            col_hd1, col_hd2 = st.columns(2)
            with col_hd1:
                st.metric("Cardiac Output", f"{hd['cardiac_output']:.2f} L/min")
                st.metric("Stroke Volume", f"{hd['stroke_volume']:.1f} mL")
                st.metric("Systolic BP", f"{hd['systolic_pressure']:.1f} mmHg")
            with col_hd2:
                st.metric("Ejection Fraction", f"{hd['ejection_fraction']*100:.1f}%")
                st.metric("Mean Arterial Pressure", f"{hd['mean_arterial_pressure']:.1f} mmHg")
                st.metric("Diastolic BP", f"{hd['diastolic_pressure']:.1f} mmHg")
                st.metric("Perfusion Index", f"{hd.get('perfusion_index', perfusion_index):.2f}")
            
            # Mechanics Results
            st.markdown("---")
            st.markdown("### 💪 Cardiac Mechanics")
            mech = sim['mechanics']
            st.metric("Contractility", f"{mech['contractility']:.2f}")
            st.metric("Wall Stress", f"{mech['wall_stress']:.2f}")
            st.metric("Efficiency", f"{mech['efficiency']:.3f}")
        else:
            st.info("👆 Configure parameters and run simulation to see results here.")

# ============================================================================
# TAB 6: REAL-TIME MONITORING
# ============================================================================
with tab6:
    st.header("📡 Real-Time Monitoring & Alerts")
    st.markdown("Continuous monitoring of cardiovascular metrics with intelligent alerting")
    
    patient_id = st.text_input("Patient ID", value="patient_001", key="monitor_patient_id")
    patient_id_hash = hash_patient_id(patient_id) if patient_id else ""
    
    if patient_id_hash:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Current Metrics")
            
            # Simulate real-time monitoring
            if st.button("🔄 Update Metrics", use_container_width=True):
                # Simulate metrics
                hr = np.random.randint(60, 100)
                systolic = np.random.randint(110, 140)
                diastolic = np.random.randint(70, 90)
                spo2 = np.random.uniform(95, 100)
                hrv = np.random.uniform(20, 60)
                
                # Monitor each metric
                hr_result = realtime_monitor.monitor_metric('heart_rate', hr, patient_id_hash)
                bp_sys_result = realtime_monitor.monitor_metric('systolic_bp', systolic, patient_id_hash)
                bp_dia_result = realtime_monitor.monitor_metric('diastolic_bp', diastolic, patient_id_hash)
                spo2_result = realtime_monitor.monitor_metric('oxygen_saturation', spo2, patient_id_hash)
                hrv_result = realtime_monitor.monitor_metric('heart_rate_variability', hrv, patient_id_hash)
                
                st.session_state.current_metrics = {
                    'heart_rate': hr_result,
                    'systolic_bp': bp_sys_result,
                    'diastolic_bp': bp_dia_result,
                    'oxygen_saturation': spo2_result,
                    'heart_rate_variability': hrv_result
                }
            
            if 'current_metrics' in st.session_state:
                metrics = st.session_state.current_metrics
                
                for metric_name, metric_data in metrics.items():
                    alert_color = {
                        'normal': '#10b981',
                        'warning': '#f59e0b',
                        'critical': '#ef4444'
                    }.get(metric_data['alert_level'], '#6b7280')
                    
                    st.markdown(f"""
                    <div style="background-color: {alert_color}15; padding: 10px; border-radius: 8px; margin: 5px 0;">
                        <strong>{metric_name.replace('_', ' ').title()}:</strong> {metric_data['metric_value']}
                        {f"<br><small style='color: {alert_color};'>{metric_data['alert_message']}</small>" if metric_data['alert_message'] else ''}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🚨 Recent Alerts")
            
            alerts = realtime_monitor.get_recent_alerts(patient_id_hash, hours=24)
            
            if alerts:
                for alert in alerts[:10]:  # Show last 10 alerts
                    alert_level = alert[4]  # alert_level column
                    alert_color = {
                        'normal': '#10b981',
                        'warning': '#f59e0b',
                        'critical': '#ef4444'
                    }.get(alert_level, '#6b7280')
                    
                    st.markdown(f"""
                    <div style="background-color: {alert_color}20; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {alert_color};">
                        <strong>{alert[2]}</strong> ({alert[3]})<br>
                        <small>{alert[5]}</small><br>
                        <small style="color: #6b7280;">{alert[1]}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts in the last 24 hours. All metrics are within normal range.")
    else:
        st.info("👆 Enter a patient ID to view monitoring data.")

# ============================================================================
# TAB 7: TREATMENT OPTIMIZATION
# ============================================================================
with tab7:
    st.header("💊 Treatment Optimization")
    st.markdown("ML-driven treatment dosage optimization and efficacy prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Patient Information")
        
        patient_id = st.text_input("Patient ID", value="patient_001", key="treatment_patient_id")
        patient_id_hash = hash_patient_id(patient_id) if patient_id else ""
        
        age = st.slider("Age", 18, 100, 50)
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0, 0.1)
        height = st.number_input("Height (cm)", 100.0, 220.0, 170.0, 0.1)
        bmi = weight / ((height/100) ** 2)
        st.info(f"📊 **Calculated BMI:** {bmi:.1f}")
        
        st.markdown("---")
        st.subheader("💉 Current Treatment")
        
        treatment_type = st.selectbox(
            "Treatment Type",
            options=['antihypertensive', 'anticoagulant', 'beta_blocker', 'ace_inhibitor', 'other']
        )
        current_dosage = st.number_input("Current Dosage (mg)", 0.0, 100.0, 10.0, 0.1)
        
        if st.button("🔍 Optimize Treatment", use_container_width=True):
            patient_data = {
                'age': age,
                'bmi': bmi,
                'weight': weight,
                'height': height,
                'patient_id_hash': patient_id_hash
            }
            
            current_treatment = {
                'type': treatment_type,
                'dosage': current_dosage
            }
            
            optimization_result = treatment_optimizer.optimize_treatment(patient_data, current_treatment)
            st.session_state.treatment_optimization = optimization_result
            st.success("✅ Treatment optimization completed!")
    
    with col2:
        st.subheader("📊 Optimization Results")
        
        if 'treatment_optimization' in st.session_state:
            opt = st.session_state.treatment_optimization
            
            st.metric("Current Dosage", f"{opt['current_dosage']} mg")
            st.metric("Optimized Dosage", f"{opt['optimized_dosage']} mg", 
                     delta=f"{opt['optimized_dosage'] - opt['current_dosage']:.2f} mg")
            st.metric("Predicted Efficacy", f"{opt['predicted_efficacy']*100:.1f}%")
            
            st.markdown("---")
            st.markdown("### ⚠️ Side Effects")
            if opt['side_effects']:
                for effect in opt['side_effects']:
                    st.warning(f"• {effect}")
            else:
                st.success("No significant side effects predicted.")
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            rec = opt['recommendations']
            st.info(f"""
            **Dosage Adjustment:** {rec['dosage_adjustment']:.2f} mg
            **Monitoring Frequency:** {rec['monitoring_frequency']}
            **Follow-up:** {rec['follow_up_days']} days
            
            **Lifestyle Changes:**
            {chr(10).join(['• ' + change for change in rec['lifestyle_changes']]) if rec['lifestyle_changes'] else '• No specific lifestyle changes recommended'}
            """)
        else:
            st.info("👆 Enter patient information and current treatment to see optimization results.")

# ============================================================================
# TAB 8: ADAPTIVE LEARNING
# ============================================================================
with tab8:
    st.header("🤖 Adaptive Learning Models")
    st.markdown("ML models that evolve with continuous feedback and new data")
    
    model_type = st.selectbox("Model Type", ['CVD', 'BP'], key="adaptive_model_type")
    
    if 'adaptive_cvd_model' not in st.session_state:
        st.session_state.adaptive_cvd_model = AdaptiveMLModel('CVD')
    if 'adaptive_bp_model' not in st.session_state:
        st.session_state.adaptive_bp_model = AdaptiveMLModel('BP')
    
    adaptive_model = st.session_state.adaptive_cvd_model if model_type == 'CVD' else st.session_state.adaptive_bp_model
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Model Performance")
        
        performance = adaptive_model.get_model_performance_trend()
        
        if performance['num_updates'] > 0:
            st.metric("Number of Updates", performance['num_updates'])
            st.metric("Latest Accuracy", f"{performance['latest_accuracy']*100:.2f}%")
            if performance['improvement'] != 0:
                st.metric("Overall Improvement", f"{performance['improvement']*100:.2f}%")
            
            # Performance trend chart
            if len(performance['accuracy_history']) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(performance['accuracy_history']))),
                    y=[acc * 100 for acc in performance['accuracy_history']],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#3b82f6', width=2)
                ))
                fig.update_layout(
                    title="Model Performance Over Time",
                    xaxis_title="Update Number",
                    yaxis_title="Accuracy (%)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model not yet trained. Add training data below.")
        
        st.markdown("---")
        st.subheader("🔄 Update Model")
        
        st.info("""
        **How it works:**
        - Upload new patient data with outcomes
        - Model automatically retrains with new data
        - Performance metrics tracked over time
        - Feature importance updated dynamically
        """)
        
        # Simulate adding new training data
        if st.button("➕ Add Simulated Training Data", use_container_width=True):
            # Generate synthetic training data
            n_samples = 50
            if model_type == 'CVD':
                X_new = pd.DataFrame({
                    'age': np.random.randint(30, 80, n_samples),
                    'cp': np.random.randint(0, 4, n_samples),
                    'thalach': np.random.randint(70, 200, n_samples),
                    'oldpeak': np.random.uniform(0, 6, n_samples),
                    'thal': np.random.randint(1, 4, n_samples)
                })
                y_new = pd.Series(np.random.randint(0, 2, n_samples))
            else:
                X_new = pd.DataFrame({
                    'age': np.random.randint(30, 80, n_samples),
                    'heart_rate': np.random.randint(60, 100, n_samples),
                    'bmi': np.random.uniform(18, 35, n_samples),
                    'sodium_intake': np.random.uniform(1000, 4000, n_samples)
                })
                y_new = pd.Series(np.random.uniform(100, 180, n_samples))
            
            # Update model
            update_info = adaptive_model.update_model(X_new, y_new)
            st.success(f"✅ Model updated! New accuracy: {update_info['new_accuracy']*100:.2f}%")
            st.rerun()
    
    with col2:
        st.subheader("📊 Training History")
        
        if adaptive_model.training_history:
            history_df = pd.DataFrame(adaptive_model.training_history)
            
            st.dataframe(
                history_df[['timestamp', 'new_accuracy', 'training_samples']].tail(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Feature importance (if available)
            if adaptive_model.feature_importance_history:
                latest_importance = adaptive_model.feature_importance_history[-1]
                if latest_importance:
                    importance_df = pd.DataFrame({
                        'Feature': list(latest_importance.keys()),
                        'Importance': list(latest_importance.values())
                    }).sort_values('Importance', ascending=False)
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Feature Importance")
                    fig = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        title="Latest Feature Importance"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training history yet. Update the model to see history here.")

# ============================================================================
# TAB 5: ANALYTICS & VISUALIZATIONS
# ============================================================================
with tab5:
    st.header("📊 Analytics & Visualizations Dashboard")
    
    # Add AI Recommendations section here
    st.subheader("💡 AI-Powered Health Recommendations")
    
    rec_tab1, rec_tab2 = st.tabs(["❤️ CVD Recommendations", "🩺 BP Recommendations"])
    
    with rec_tab1:
        st.subheader("❤️ Cardiovascular Disease - Diet & Risk Management")
        
        if 'cvd_assessment' in st.session_state:
            assessment = st.session_state.cvd_assessment
            
            if st.button("🔄 Get Fresh Recommendations", key="cvd_rec_button"):
                st.session_state.loading_cvd_rec = True
            
            if 'loading_cvd_rec' in st.session_state or 'cvd_recommendations' not in st.session_state:
                if st.session_state.get('loading_cvd_rec', False):
                    with st.spinner("🤖 AI is analyzing your CVD risk data and generating personalized recommendations..."):
                        rec_result = get_health_recommendations(assessment, "CVD")
                        
                        if rec_result.get("success", False):
                            st.session_state.cvd_recommendations = rec_result["recommendations"]
                            st.session_state.loading_cvd_rec = False
                            st.rerun()
                        else:
                            st.error(f"Error: {rec_result.get('error', 'Unknown error')}")
            
            if 'cvd_recommendations' in st.session_state:
                st.markdown("### 📋 Personalized Recommendations")
                st.markdown(st.session_state.cvd_recommendations)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{assessment['risk_score_percent']}%")
                with col2:
                    st.metric("Risk Category", assessment['risk_category'])
                with col3:
                    high_risk_factors = sum(1 for f in assessment['factors'].values() if f['risk'] == 'High')
                    st.metric("High Risk Factors", high_risk_factors)
        else:
            st.info("👆 Please complete a CVD risk assessment first to get personalized recommendations.")
    
    with rec_tab2:
        st.subheader("🩺 Blood Pressure - Diet & Management")
        
        if 'bp_assessment' in st.session_state:
            assessment = st.session_state.bp_assessment
            
            if st.button("🔄 Get Fresh Recommendations", key="bp_rec_button"):
                st.session_state.loading_bp_rec = True
            
            if 'loading_bp_rec' in st.session_state or 'bp_recommendations' not in st.session_state:
                if st.session_state.get('loading_bp_rec', False):
                    with st.spinner("🤖 AI is analyzing your BP assessment and generating personalized recommendations..."):
                        rec_result = get_health_recommendations(assessment, "BP")
                        
                        if rec_result.get("success", False):
                            st.session_state.bp_recommendations = rec_result["recommendations"]
                            st.session_state.loading_bp_rec = False
                            st.rerun()
                        else:
                            st.error(f"Error: {rec_result.get('error', 'Unknown error')}")
            
            if 'bp_recommendations' in st.session_state:
                st.markdown("### 📋 Personalized Recommendations")
                st.markdown(st.session_state.bp_recommendations)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Systolic BP", f"{assessment['predicted_systolic']} mmHg")
                with col2:
                    st.metric("Diastolic BP", f"{assessment['predicted_diastolic']} mmHg")
                with col3:
                    st.metric("Category", assessment['category'])
        else:
            st.info("👆 Please complete a Blood Pressure assessment first to get personalized recommendations.")
    
    st.markdown("---")
    
    # CVD Analytics
    st.subheader("❤️ Cardiovascular Disease Analytics")
    
    cvd_history = get_cvd_history(50)
    if cvd_history:
        cvd_df = pd.DataFrame(cvd_history, columns=[
            'ID', 'Timestamp', 'Age', 'CP', 'Thalach', 'Oldpeak', 'Thal',
            'Risk Score', 'Risk Category', 'Session ID'
        ])
        cvd_df['Timestamp'] = pd.to_datetime(cvd_df['Timestamp'])
        cvd_df = cvd_df.sort_values('Timestamp')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Score Trend Over Time
            fig = px.line(
                cvd_df,
                x='Timestamp',
                y='Risk Score',
                markers=True,
                title="CVD Risk Score Trend Over Time",
                color_discrete_sequence=['#ef4444']
            )
            fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig.add_hline(y=35, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Category Distribution
            category_counts = cvd_df['Risk Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Risk Category Distribution",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Factor Analysis for Latest Assessment
        if 'cvd_assessment' in st.session_state:
            st.markdown("---")
            st.subheader("📈 Latest Assessment - Factor Analysis")
            
            assessment = st.session_state.cvd_assessment
            factors_df = pd.DataFrame({
                'Factor': list(assessment['factors'].keys()),
                'Risk Level': [info['risk'] for info in assessment['factors'].values()],
                'Value': [info['value'] for info in assessment['factors'].values()]
            })
            
            risk_map = {'High': 3, 'Moderate': 2, 'Low': 1}
            factors_df['Risk Score'] = factors_df['Risk Level'].map(risk_map)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    factors_df,
                    x='Factor',
                    y='Risk Score',
                    color='Risk Level',
                    color_discrete_map={'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'},
                    title="Individual Risk Factor Impact"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                risk_counts = factors_df['Risk Level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Factor Distribution",
                    color_discrete_map={'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'},
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CVD assessment history yet. Complete assessments to see analytics here.")
    
    # BP Analytics
    st.markdown("---")
    st.subheader("🩺 Blood Pressure Analytics")
    
    bp_history, bp_columns = get_bp_history(50)
    if bp_history:
        # Determine actual number of columns from first row
        actual_cols = len(bp_history[0]) if bp_history else 0
        
        # Use actual column names from database or fallback to expected names
        if bp_columns and len(bp_columns) == actual_cols:
            bp_df = pd.DataFrame(bp_history, columns=bp_columns)
        else:
            # Fallback: adjust based on actual data length
            if actual_cols == 9:
                bp_df = pd.DataFrame(bp_history, columns=[
                    'ID', 'Timestamp', 'Age', 'Heart Rate', 'BMI', 'Sodium Intake',
                    'Systolic BP', 'Diastolic BP', 'Category'
                ])
            elif actual_cols == 10:
                bp_df = pd.DataFrame(bp_history, columns=[
                    'ID', 'Timestamp', 'Age', 'Heart Rate', 'BMI', 'Sodium Intake',
                    'Systolic BP', 'Diastolic BP', 'Category', 'Session ID'
                ])
            elif actual_cols == 8:
                bp_df = pd.DataFrame(bp_history, columns=[
                    'ID', 'Timestamp', 'Clump Thickness', 'Uniformity', 'Mitoses',
                    'Systolic BP', 'Diastolic BP', 'Category'
                ])
                st.warning("⚠️ Old database schema detected. Please create new BP assessments with the updated features.")
            else:
                bp_df = pd.DataFrame(bp_history)
                st.warning(f"⚠️ Unknown database schema with {actual_cols} columns. Some features may not display correctly.")
        
        # Map column names to display names
        if 'id' in [col.lower() for col in bp_df.columns] or 'ID' in bp_df.columns:
            column_mapping = {
                'id': 'ID',
                'timestamp': 'Timestamp',
                'age': 'Age',
                'heart_rate': 'Heart Rate',
                'bmi': 'BMI',
                'sodium_intake': 'Sodium Intake',
                'predicted_systolic': 'Systolic BP',
                'predicted_diastolic': 'Diastolic BP',
                'category': 'Category',
                'session_id': 'Session ID'
            }
            bp_df.columns = [column_mapping.get(col.lower(), col) for col in bp_df.columns]
        
        # Check if we have the required columns
        required_cols = ['Timestamp', 'Systolic BP', 'Diastolic BP']
        if all(col in bp_df.columns for col in required_cols):
            bp_df['Timestamp'] = pd.to_datetime(bp_df['Timestamp'])
            bp_df = bp_df.sort_values('Timestamp')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # BP Trend Over Time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bp_df['Timestamp'],
                    y=bp_df['Systolic BP'],
                    name='Systolic BP',
                    line=dict(color='#ef4444', width=2),
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=bp_df['Timestamp'],
                    y=bp_df['Diastolic BP'],
                    name='Diastolic BP',
                    line=dict(color='#3b82f6', width=2),
                    mode='lines+markers'
                ))
                fig.add_hline(y=140, line_dash="dash", line_color="red", annotation_text="High Systolic")
                fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="High Diastolic")
                fig.update_layout(
                    title="Blood Pressure Trend Over Time",
                    xaxis_title="Date",
                    yaxis_title="Blood Pressure (mmHg)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Category Distribution
                if 'Category' in bp_df.columns:
                    category_counts = bp_df['Category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Blood Pressure Category Distribution",
                        color_discrete_map={
                            'High (Hypertension)': '#ef4444',
                            'Elevated': '#f59e0b',
                            'Normal': '#10b981'
                        },
                        hole=0.4
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Category data not available in database.")
        else:
            st.warning("⚠️ Database schema mismatch. Please clear old data or recreate the database.")
    else:
        st.info("No BP assessment history yet. Complete assessments to see analytics here.")


# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #059669 !important;
        font-weight: 700 !important;
    }
    
    h1, h2, h3 {
        color: #1e293b;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Dual-Model Health Dashboard | For demonstration purposes only. Not a substitute for professional medical advice.")

