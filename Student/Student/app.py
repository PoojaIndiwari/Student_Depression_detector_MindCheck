"""
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import glob

# Page configuration with professional theme
st.set_page_config(
    page_title="MindCheck | Student Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with clean design and black text
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------------- GLOBAL ---------------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

/* Remove top padding */
.block-container {
    padding-top: 1.5rem;
}

/* ---------------- HEADERS ---------------- */
h1, h2, h3, h4, h5 {
    color: #0f172a;
    font-weight: 600;
}

p, label, span {
    color: #334155;
}

/* ---------------- CARDS ---------------- */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.5rem;
}

/* ---------------- HEADER BANNER ---------------- */
.header-section {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
}

.header-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #e0e7ff;
}

/* ---------------- INPUT FIELDS ---------------- */
input, textarea, select {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}

/* Number input buttons */
button[kind="secondary"] {
    color: #ffffff !important;
}

/* Dropdown text */
.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Slider labels */
.stSlider label {
    color: #0f172a !important;
}

/* Slider value */
.stSlider div[data-testid="stTickBar"] {
    color: #0f172a;
}

/* ---------------- RADIO / CHECKBOX ---------------- */
.stRadio label, .stCheckbox label {
    color: #0f172a !important;
}

/* ---------------- BUTTON ---------------- */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
    color: #ffffff;
    border-radius: 10px;
    padding: 0.8rem 2.2rem;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #3730a3);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
}

/* ---------------- RESULT CARDS ---------------- */
.result-card {
    border-radius: 14px;
    padding: 2rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
}

.low-risk {
    border-left: 6px solid #16a34a;
    background: #f0fdf4;
}

.moderate-risk {
    border-left: 6px solid #f59e0b;
    background: #fffbeb;
}

.high-risk {
    border-left: 6px solid #dc2626;
    background: #fef2f2;
}

/* ---------------- BADGES ---------------- */
.risk-badge {
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.low-badge {
    background: #dcfce7;
    color: #166534;
}

.moderate-badge {
    background: #ffedd5;
    color: #9a3412;
}

.high-badge {
    background: #fee2e2;
    color: #991b1b;
}

/* ---------------- INFO / ALERT BOXES ---------------- */
.info-box {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 1.4rem;
}

.warning-box {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 1.4rem;
}

.critical-box {
    background: #fef2f2;
    border-left: 5px solid #dc2626;
    border-radius: 12px;
    padding: 1.4rem;
}

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

[data-testid="stSidebar"] h3 {
    color: #0f172a;
}

[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] li {
    color: #334155;
}

/* ---------------- LINKS ---------------- */
a {
    color: #2563eb !important;
    font-weight: 500;
}

/* ---------------- METRICS ---------------- */
[data-testid="stMetricValue"] {
    color: #4f46e5;
    font-size: 1.8rem;
}

[data-testid="stMetricLabel"] {
    color: #475569;
}
</style>
""", unsafe_allow_html=True)


class DepressionPredictor:
    """Enhanced prediction class with better error handling"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.encoding_map = None
        self.model_loaded = False
        self.model_path = None
        self.model_name = "Unknown"

    def find_model_file(self):
        """Smart model file detection"""
        search_paths = [
            'models/best_model.pkl',
            'evaluation_results/models/best_model.pkl',
            'models/best_model_*.pkl',
        ]

        for path in search_paths:
            if '*' in path:
                files = glob.glob(path)
                if files:
                    return files[0]
            elif os.path.exists(path):
                return path

        return None

    def load_artifacts(self):
        """Load model and preprocessing artifacts with enhanced feedback"""
        try:
            self.model_path = self.find_model_file()

            required_files = {
                'scaler': 'data/processed/scaler.pkl',
                'features': 'data/processed/feature_names.json',
                'encoding': 'data/processed/encoding_map.json'
            }

            missing = []
            if not self.model_path:
                missing.append("Model file (.pkl)")

            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing.append(f"{path}")

            if missing:
                st.error("### Missing Required Files")
                for file in missing:
                    st.error(file)
                st.info("**Solution:** Run the training script first to generate these files.")
                return False

            # Load artifacts
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load('data/processed/scaler.pkl')

            with open('data/processed/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)

            with open('data/processed/encoding_map.json', 'r') as f:
                self.encoding_map = json.load(f)

            # Extract model name
            self.model_name = type(self.model).__name__
            self.model_loaded = True
            return True

        except Exception as e:
            st.error(f"### Error Loading Model")
            st.exception(e)
            return False

    def create_input_template(self):
        return pd.DataFrame(np.zeros((1, len(self.feature_names))), columns=self.feature_names)

    def process_input(self, input_dict):
        """Process user input with proper encoding"""
        template = self.create_input_template()

        # Identify numeric features
        numeric_features = [f for f in self.feature_names if not any(
            f.startswith(cat + '_') for cat in self.encoding_map.keys()
        )]

        # Fill numeric values
        for feature in numeric_features:
            if feature in input_dict:
                template.at[0, feature] = input_dict[feature]

        # Scale numeric features
        if len(numeric_features) > 0:
            template[numeric_features] = self.scaler.transform(template[numeric_features])

        # Process categorical features
        for original_feature, encoded_columns in self.encoding_map.items():
            if original_feature in input_dict:
                input_value = str(input_dict[original_feature]).strip().title()
                for encoded_col in encoded_columns:
                    if input_value in encoded_col:
                        template.at[0, encoded_col] = 1
                        break
        return template

    def predict(self, input_dict):
        """Make prediction with probability"""
        if not self.model_loaded:
            return None, None

        processed_data = self.process_input(input_dict)
        prediction = self.model.predict(processed_data)[0]

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)[0]
            probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            probability = 0.5 if prediction == 0 else 1.0

        return prediction, probability


@st.cache_resource
def get_predictor():
    """Initialize and cache predictor"""
    predictor = DepressionPredictor()
    predictor.load_artifacts()
    return predictor


def get_risk_assessment(probability):
    """Enhanced risk level assessment"""
    if probability >= 0.8:
        return {
            'level': 'CRITICAL',
            'class': 'high-risk',
            'badge_class': 'critical-badge',
            'color': '#dc2626',
            'recommendation': 'Immediate professional help strongly recommended',
            'actions': [
                'Contact a mental health professional today',
                'Reach out to campus counseling services',
                'Talk to a trusted friend or family member',
                'Call crisis hotline: 988'
            ]
        }
    elif probability >= 0.6:
        return {
            'level': 'HIGH',
            'class': 'high-risk',
            'badge_class': 'high-badge',
            'color': '#d97706',
            'recommendation': 'Professional consultation recommended',
            'actions': [
                'Schedule appointment with counselor',
                'Practice stress management techniques',
                'Maintain regular sleep schedule',
                'Connect with support groups'
            ]
        }
    elif probability >= 0.4:
        return {
            'level': 'MODERATE',
            'class': 'moderate-risk',
            'badge_class': 'moderate-badge',
            'color': '#2563eb',
            'recommendation': 'Monitor symptoms and practice self-care',
            'actions': [
                'Track your mood daily',
                'Maintain healthy routines',
                'Stay connected with friends',
                'Consider talking to a counselor'
            ]
        }
    else:
        return {
            'level': 'LOW',
            'class': 'low-risk',
            'badge_class': 'low-badge',
            'color': '#059669',
            'recommendation': 'Maintain healthy habits',
            'actions': [
                'Continue current wellness practices',
                'Stay physically active',
                'Maintain social connections',
                'Practice mindfulness regularly'
            ]
        }


def create_modern_gauge(probability):
    """Create modern gauge chart"""
    risk = get_risk_assessment(probability)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 20, 'color': '#000000', 'family': 'Inter'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': risk['color']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#000000"},
            'bar': {'color': risk['color'], 'thickness': 0.5},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#d1fae5'},
                {'range': [40, 60], 'color': '#fef3c7'},
                {'range': [60, 80], 'color': '#fed7aa'},
                {'range': [80, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': risk['color'], 'width': 4},
                'thickness': 0.6,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': '#000000'}
    )
    return fig


def create_feature_radar(input_dict):
    """Create radar chart for features"""
    categories = []
    values = []

    # Normalize values to 0-5 scale
    for key, value in input_dict.items():
        if key in ['Academic Pressure', 'Work Pressure', 'Financial Stress', 'Study Satisfaction']:
            categories.append(key)
            values.append(value)

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.1)',
        line=dict(color='rgb(79, 70, 229)', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], color='#000000'),
            angularaxis=dict(color='#000000'),
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 11, 'color': '#000000'}
    )
    return fig


def main():
    # Header Section
    st.markdown("""
        <div class='header-section'>
            <div class='header-title'>MindCheck</div>
            <div class='header-subtitle'>AI-Powered Student Mental Health Assessment</div>
            <p style='color: #e0e7ff; font-size: 0.95rem; margin-top: 1rem;'>
                Early detection • Professional insights • Compassionate care
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load predictor
    predictor = get_predictor()
    if not predictor.model_loaded:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")
        st.success(f"Model: {predictor.model_name}")
        st.info(f"Features: {len(predictor.feature_names)}")

        st.markdown("---")
        st.markdown("### About This Tool")
        st.markdown("""
        This AI-powered tool screens for depression risk of students based on:
        - Academic factors
        - Lifestyle habits
        - Personal circumstances

        **Note:** This is not a medical diagnosis. Always consult healthcare professionals.
        """)

        st.markdown("---")
        st.markdown("### Statistics")
        if os.path.exists('data/processed/preprocessed_data.csv'):
            df = pd.read_csv('data/processed/preprocessed_data.csv')
            st.metric("Training Samples", len(df))
            st.metric("Model Accuracy", "~85%")

    # Main Tabs
    tabs = st.tabs(["Individual Assessment", "Batch Analysis", "Model Insights"])

    # TAB 1: Individual Assessment
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Student Assessment Form")
        st.markdown("Please provide accurate information for the best assessment.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Academic Factors")
            academic_pressure = st.slider(
                "Academic Pressure",
                1, 5, 3,
                help="How much pressure do you feel from academic demands?"
            )
            work_pressure = st.slider(
                "Work/Job Pressure",
                1, 5, 2,
                help="Pressure from work or job responsibilities"
            )
            cgpa = st.number_input(
                "CGPA / GPA",
                0.0, 10.0, 7.5, 0.1,
                help="Your current grade point average"
            )
            study_satisfaction = st.slider(
                "Study Satisfaction",
                1, 5, 3,
                help="How satisfied are you with your studies?"
            )

        with col2:
            st.markdown("#### Lifestyle & Wellness")
            financial_stress = st.slider(
                "Financial Stress",
                1, 5, 2,
                help="Level of financial burden or worry"
            )
            sleep_duration = st.selectbox(
                "Average Sleep Duration",
                ['Less Than 5 Hours', '5-6 Hours', '7-8 Hours', 'More Than 8 Hours'],
                index=2,
                help="How many hours do you typically sleep?"
            )
            dietary_habits = st.selectbox(
                "Dietary Habits",
                ['Healthy', 'Moderate', 'Unhealthy'],
                index=1,
                help="Overall quality of your diet"
            )

        with col3:
            st.markdown("#### Personal Information")
            age = st.number_input(
                "Age",
                16, 40, 21,
                help="Your current age"
            )
            gender = st.selectbox(
                "Gender",
                ['Male', 'Female', 'Other', 'Prefer not to say'],
                help="Your gender identity"
            )
            work_study_hours = st.number_input(
                "Daily Work/Study Hours",
                0, 24, 8,
                help="Total hours spent on work and study per day"
            )

            st.markdown("#### Critical Assessment")
            suicidal_thoughts = st.radio(
                "Have you experienced suicidal thoughts?",
                ['No', 'Yes'],
                help="This information is crucial for risk assessment"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Predict Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "Analyze Mental Health Risk",
                type="primary",
                use_container_width=True
            )

        if predict_button:
            student_input = {
                'Academic Pressure': academic_pressure,
                'Work Pressure': work_pressure,
                'Financial Stress': financial_stress,
                'CGPA': cgpa,
                'Study Satisfaction': study_satisfaction,
                'Sleep Duration': sleep_duration,
                'Dietary Habits': dietary_habits,
                'Age': age,
                'Gender': gender,
                'Work/Study Hours': work_study_hours,
                'Suicidal Thoughts': suicidal_thoughts
            }

            # CRITICAL SAFETY CHECK
            if suicidal_thoughts == 'Yes':
                st.markdown("""
                    <div class='critical-box'>
                        <h3 style='color: #000000; margin-top: 0;'>Crisis Support Available Now</h3>
                        <p style='font-size: 1rem; color: #000000; font-weight: 600;'>
                            You are not alone. Help is available 24/7.
                        </p>
                        <div style='background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                            <h4 style='color: #000000;'>Immediate Contact:</h4>
                            <ul style='font-size: 1rem; line-height: 1.8; color: #000000;'>
                                <li><strong>National Suicide Prevention Lifeline:</strong> <a href="tel:988" style='color: #dc2626; font-weight: bold;'>988</a></li>
                                <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
                                <li><strong>Online Chat:</strong> <a href="https://988lifeline.org/chat/" target="_blank">988lifeline.org/chat</a></li>
                            </ul>
                        </div>
                        <p style='margin-top: 1rem; font-size: 0.9rem; color: #000000;'>
                            <strong>If you are in immediate danger, please call 911 or go to your nearest emergency room.</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Still continue with analysis but mark as critical
                st.warning("Continuing with assessment, but please seek immediate professional help.")

            with st.spinner('Analyzing your data...'):
                prediction, probability = predictor.predict(student_input)

                # Override risk if suicidal thoughts present
                if suicidal_thoughts == 'Yes':
                    probability = max(probability, 0.95)  # Ensure very high risk
                    prediction = 1

                risk = get_risk_assessment(probability)

            # Results Section
            st.markdown("---")
            st.markdown("## Assessment Results")

            # Result Display
            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.plotly_chart(create_modern_gauge(probability), use_container_width=True)

            with result_col2:
                status = "Depression Risk Detected" if prediction == 1 else "Low Depression Risk"
                confidence = probability if prediction == 1 else (1 - probability)

                # Add suicidal thoughts warning
                if suicidal_thoughts == 'Yes':
                    st.markdown("""
                        <div class='result-card high-risk' style='border: 2px solid #dc2626;'>
                            <div class='result-title' style='color: #dc2626;'>CRITICAL RISK - IMMEDIATE ATTENTION NEEDED</div>
                            <div class='result-subtitle' style='color: #374151;'>Suicidal Thoughts Reported</div>
                            <div class='confidence-score' style='color: #dc2626;'>URGENT</div>
                            <p style='color: #374151; margin-top: 1rem; font-weight: 600;'>
                                Please contact crisis services immediately (988)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='result-card {risk['class']}'>
                            <span class='risk-badge {risk['badge_class']}' style='margin-bottom: 1rem; display: inline-block;'>{risk['level']} RISK</span>
                            <div class='result-title' style='color: #000000;'>{status}</div>
                            <div class='confidence-score'>{confidence:.1%}</div>
                            <p style='color: #374151; margin-top: 0.5rem;'>Confidence Score</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown(f"""
                    <div class='{"warning-box" if prediction == 1 else "success-box"}'>
                        <h4 style='margin-top: 0; color: #000000;'>Recommendation</h4>
                        <p style='font-size: 1rem; margin-bottom: 0; color: #000000;'>{risk['recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)

            with rec_col2:
                st.markdown("""
                    <div class='info-box'>
                        <h4 style='margin-top: 0; color: #000000;'>Suggested Actions</h4>
                    </div>
                """, unsafe_allow_html=True)
                for action in risk['actions']:
                    st.markdown(f"- {action}")

            # Visualization
            st.markdown("---")
            st.markdown("### Profile Analysis")

            viz_col1, viz_col2 = st.columns([1, 1])

            with viz_col1:
                st.plotly_chart(create_feature_radar(student_input), use_container_width=True)

            with viz_col2:
                # Factor breakdown
                factors = pd.DataFrame({
                    'Factor': ['Academic', 'Work', 'Financial', 'Satisfaction'],
                    'Score': [academic_pressure, work_pressure, financial_stress, 6 - study_satisfaction]
                })
                fig = px.bar(
                    factors,
                    x='Score',
                    y='Factor',
                    orientation='h',
                    color='Score',
                    color_continuous_scale=['#059669', '#d97706', '#dc2626'],
                    title='Stress Factor Breakdown'
                )
                fig.update_layout(
                    showlegend=False,
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#000000'},
                    title_font_color='#000000',
                    xaxis_title_font_color='#000000',
                    yaxis_title_font_color='#000000'
                )
                st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Batch Analysis
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Batch Assessment")
        st.markdown("Upload a CSV file containing multiple student records for batch analysis.")

        st.markdown("""
            <div class='info-box'>
                <h4 style='color: #000000;'>Required CSV Columns:</h4>
                <p style='color: #000000;'>
                Academic Pressure, Work Pressure, Financial Stress, CGPA, Study Satisfaction, 
                Sleep Duration, Dietary Habits, Age, Work/Study Hours
                </p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose CSV File", type=['csv'])

        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} records")

                with st.expander("Preview Data"):
                    st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Run Batch Analysis", type="primary", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, row in batch_df.iterrows():
                        status_text.text(f"Processing student {idx + 1} of {len(batch_df)}...")
                        pred, prob = predictor.predict(row.to_dict())
                        risk = get_risk_assessment(prob)

                        results.append({
                            'Student_ID': idx + 1,
                            'Prediction': 'At Risk' if pred == 1 else 'Low Risk',
                            'Risk_Level': risk['level'],
                            'Confidence': f"{prob:.1%}",
                            'Recommendation': risk['recommendation']
                        })
                        progress_bar.progress((idx + 1) / len(batch_df))

                    status_text.text("Analysis Complete!")
                    results_df = pd.DataFrame(results)

                    # Summary Statistics
                    st.markdown("### Batch Summary")
                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                    at_risk = (results_df['Prediction'] == 'At Risk').sum()
                    low_risk = (results_df['Prediction'] == 'Low Risk').sum()
                    high_risk = (results_df['Risk_Level'] == 'HIGH').sum() + (
                            results_df['Risk_Level'] == 'CRITICAL').sum()

                    with sum_col1:
                        st.markdown(f"""
                            <div class='stat-card'>
                                <div class='stat-value'>{len(batch_df)}</div>
                                <div class='stat-label'>Total Students</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with sum_col2:
                        st.markdown(f"""
                            <div class='stat-card'>
                                <div class='stat-value' style='color: #dc2626;'>{at_risk}</div>
                                <div class='stat-label'>At Risk</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with sum_col3:
                        st.markdown(f"""
                            <div class='stat-card'>
                                <div class='stat-value' style='color: #059669;'>{low_risk}</div>
                                <div class='stat-label'>Low Risk</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with sum_col4:
                        st.markdown(f"""
                            <div class='stat-card'>
                                <div class='stat-value' style='color: #d97706;'>{high_risk}</div>
                                <div class='stat-label'>Urgent Cases</div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Results Table
                    st.markdown("### Detailed Results")
                    st.dataframe(
                        results_df.style.apply(
                            lambda x: ['background-color: #fee2e2' if v == 'At Risk' else 'background-color: #d1fae5'
                                       for v in x],
                            subset=['Prediction']
                        ),
                        use_container_width=True
                    )

                    # Download Button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        f"mental_health_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )

                    # Risk Distribution Chart
                    st.markdown("### Risk Distribution")
                    risk_dist = results_df['Risk_Level'].value_counts()
                    fig = px.pie(
                        values=risk_dist.values,
                        names=risk_dist.index,
                        title='Risk Level Distribution',
                        color_discrete_sequence=['#059669', '#d97706', '#dc2626', '#2563eb']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter', 'size': 12, 'color': '#000000'},
                        title_font_color='#000000'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV has the correct column names and data format.")

        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3: Model Insights
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Model Performance & Statistics")

        # Model Information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{predictor.model_name}</div>
                    <div class='stat-label'>Model Type</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(predictor.feature_names)}</div>
                    <div class='stat-label'>Total Features</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            if os.path.exists('data/processed/preprocessed_data.csv'):
                df = pd.read_csv('data/processed/preprocessed_data.csv')
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-value'>{len(df):,}</div>
                        <div class='stat-label'>Training Samples</div>
                    </div>
                """, unsafe_allow_html=True)

        # Model Comparison
        stats_path = 'evaluation_results/model_comparison.csv'

        if os.path.exists(stats_path):
            st.markdown("---")
            st.markdown("### Model Comparison")

            comparison_df = pd.read_csv(stats_path)

            # Metrics Grid
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            best_model = comparison_df.iloc[0]

            with metric_col1:
                st.metric("Best Accuracy", f"{best_model['Accuracy']:.3f}")
            with metric_col2:
                st.metric("F1 Score", f"{best_model['F1']:.3f}")
            with metric_col3:
                st.metric("Recall (Depressed)", f"{best_model['Recall_Depressed']:.3f}")
            with metric_col4:
                st.metric("ROC AUC", f"{best_model['ROC_AUC']:.3f}")

            # Comparison Table
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=['Accuracy', 'F1', 'Recall_Depressed', 'ROC_AUC'],
                    color='#d1fae5'
                ),
                use_container_width=True
            )

            # Visualization
            st.markdown("### Performance Comparison")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='F1',
                    title='F1 Score by Model',
                    color='F1',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#000000'},
                    title_font_color='#000000',
                    xaxis_title_font_color='#000000',
                    yaxis_title_font_color='#000000',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            with viz_col2:
                fig = px.scatter(
                    comparison_df,
                    x='Recall_Depressed',
                    y='Precision_Depressed',
                    size='F1',
                    color='Model',
                    title='Precision vs Recall',
                    hover_data=['Accuracy']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#000000'},
                    title_font_color='#000000',
                    xaxis_title_font_color='#000000',
                    yaxis_title_font_color='#000000'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Model Interpretations
            st.markdown("---")
            st.markdown("### Model Interpretation")

            if best_model['Recall_Depressed'] >= 0.7:
                st.success(f"""
                    **Good Recall ({best_model['Recall_Depressed']:.1%})**: 
                    The model successfully identifies most students at risk of depression.
                """)
            else:
                st.warning(f"""
                    **Low Recall ({best_model['Recall_Depressed']:.1%})**: 
                    The model may miss some students at risk. Consider threshold adjustment.
                """)

            if best_model['Precision_Depressed'] >= 0.7:
                st.success(f"""
                    **Good Precision ({best_model['Precision_Depressed']:.1%})**: 
                    Most positive predictions are accurate, reducing false alarms.
                """)
            else:
                st.info(f"""
                    **Moderate Precision ({best_model['Precision_Depressed']:.1%})**: 
                    Some false positives may occur. Additional screening recommended.
                """)

        else:
            st.warning("Model comparison statistics not found. Run the evaluation script first.")

        # Dataset Information
        if os.path.exists('data/processed/preprocessed_data.csv'):
            st.markdown("---")
            st.markdown("### Dataset Statistics")

            df = pd.read_csv('data/processed/preprocessed_data.csv')

            dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

            with dataset_col1:
                st.metric("Total Samples", f"{len(df):,}")
            with dataset_col2:
                st.metric("Total Features", len(df.columns))
            with dataset_col3:
                if 'Depression' in df.columns:
                    class_dist = df['Depression'].value_counts()
                    st.metric("Class Balance", f"{class_dist.min()}/{class_dist.max()}")

            # Class Distribution
            if 'Depression' in df.columns:
                st.markdown("#### Class Distribution")
                class_counts = df['Depression'].value_counts()
                fig = px.bar(
                    x=['Healthy (0)', 'Depressed (1)'],
                    y=class_counts.values,
                    title='Training Data Distribution',
                    color=class_counts.values,
                    color_continuous_scale=['#059669', '#dc2626']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#000000'},
                    showlegend=False,
                    xaxis_title='Class',
                    yaxis_title='Count',
                    title_font_color='#000000',
                    xaxis_title_font_color='#000000',
                    yaxis_title_font_color='#000000'
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #000000; padding: 2rem;'>
            <p style='font-size: 0.9rem;'>
                <strong>MindCheck</strong>
            </p>
            <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
                This tool is for screening purposes only and does not replace professional medical advice.
                <br>Always consult with qualified healthcare providers for diagnosis and treatment.
            </p>
            <p style='font-size: 0.7rem; margin-top: 1rem; opacity: 0.7;'>
                © 2025 MindCheck by Pooja | Built with Streamlit & Scikit-learn
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()