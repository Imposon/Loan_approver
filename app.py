import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loan_model import LoanPredictor
import time
import os

# Page Config
st.set_page_config(
    page_title="Altus Bank - AI Loan Officer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for that "Crazy" / Premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 15, 30) 0%, rgb(5, 5, 10) 90.2%);
        color: white;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.5);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        padding: 15px 45px;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(79, 172, 254, 0.5);
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }
    
    /* Input field styling */
    .stSelectbox, .stSlider, .stRadio {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 5px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor
@st.cache_resource
def get_predictor():
    predictor = LoanPredictor()
    if not predictor.load_model():
        predictor.train()
    return predictor

predictor = get_predictor()

# Sidebar Inputs
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🏦 ALTUS BANK")
        
    st.markdown("### 🧬 BIOMETRIC DATA")
    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    with col_s2:
        married = st.selectbox("Married", ["No", "Yes"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    st.markdown("### 💰 FINANCIAL VECTORS")
    app_income = st.slider("Applicant Income ($)", 0, 80000, 5000)
    co_income = st.slider("Co-applicant Income ($)", 0, 40000, 0)
    loan_amount = st.slider("Loan Amount ($k)", 10, 700, 150)
    loan_term = st.select_slider("Term (Months)", options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480], value=360)
    credit_history = st.radio("Credit History", [1.0, 0.0], format_func=lambda x: "Exemplary" if x == 1.0 else "Sub-par")

# Main Application
st.markdown('<h1 class="main-header">ALTUS DECISION ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; color: #a1a1a1; margin-bottom: 30px;">AI-Driven Risk Abstraction & Credit Assessment</p>', unsafe_allow_html=True)

# Application Snapshot
input_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': app_income,
    'CoapplicantIncome': co_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 RISK TOPOLOGY")
    
    # Feature Importance based on the trained model
    if predictor.model:
        importances = predictor.model.feature_importances_
        features = predictor.feature_names
        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(feat_df, y='Feature', x='Importance', orientation='h', 
                         color='Importance',
                         color_continuous_scale='Blues',
                         template="plotly_dark")
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font_color='white', height=400, margin=dict(l=20, r=20, t=20, b=20),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛡️ NEURAL AUDIT")
    
    if st.button("EXECUTE RISK ASSESSMENT"):
        with st.spinner("Decoupling financial variables..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            prediction, prob = predictor.predict(input_data)
        
        # probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            number = {'suffix': "%", 'font': {'size': 60, 'color': '#00f2fe'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00f2fe"},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 69, 58, 0.2)'},
                    {'range': [40, 75], 'color': 'rgba(255, 214, 10, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(48, 209, 88, 0.2)'}],
                'threshold': {
                    'line': {'color': "#00f2fe", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Outfit"}, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.markdown("""
                <div style="background: rgba(48, 209, 88, 0.1); border: 1px solid #30d158; border-radius: 15px; padding: 20px; text-align: center;">
                    <h2 style="color: #30d158; margin: 0;">VERDICT: APPROVED</h2>
                    <p style="color: #a1a1a1; margin: 5px 0 0 0;">Applicant satisfies all safety protocols.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: rgba(255, 69, 58, 0.1); border: 1px solid #ff453a; border-radius: 15px; padding: 20px; text-align: center;">
                    <h2 style="color: #ff453a; margin: 0;">VERDICT: DECLINED</h2>
                    <p style="color: #a1a1a1; margin: 5px 0 0 0;">Risk threshold breach detected. Collateral insufficient.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="height: 350px; display: flex; align-items: center; justify-content: center; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;">
                <p style="color: rgba(255,255,255,0.3); font-size: 1.2rem;">Ready for Audit Sequence...</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced Insights
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🪐 MULTIVERSE PROJECTION ('What-If')")
st.markdown("Adjust the slider to see how approval probability shifts across parallel financial scenarios.")

wi_col1, wi_col2 = st.columns([1, 3], gap="large")

with wi_col1:
    target_feat = st.selectbox("SENSITIVITY PIVOT", ["ApplicantIncome", "LoanAmount", "CoapplicantIncome"])
    if "Income" in target_feat:
        range_min, range_max = 0, 100000
    else:
        range_min, range_max = 10, 1000

with wi_col2:
    test_values = np.linspace(range_min, range_max, 50)
    probs = []
    
    base_input = input_data.copy()
    for v in test_values:
        base_input[target_feat] = v
        _, p = predictor.predict(base_input)
        probs.append(p)
    
    wi_df = pd.DataFrame({target_feat: test_values, 'Approval Probability': probs})
    
    fig_wi = px.area(wi_df, x=target_feat, y='Approval Probability', 
                     template="plotly_dark",
                     color_discrete_sequence=['#00f2fe'])
    fig_wi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                         font_color='white', margin=dict(l=20, r=20, t=10, b=20),
                         yaxis_range=[0, 1])
    fig_wi.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Minimum Threshold")
    st.plotly_chart(fig_wi, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align: center; color: #555; font-size: 0.8rem;">ALTUS CORP © 2026 | QUANTUM RISK DEPT</p>', unsafe_allow_html=True)
