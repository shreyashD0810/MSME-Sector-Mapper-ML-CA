# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
from io import BytesIO
import ast
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page config
st.set_page_config(
    page_title="MSME Performance Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS WITH UPDATED STYLING ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Open+Sans:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    body {
        font-family: 'Open Sans', sans-serif;
        color: #000000;
        background: #FFFFFF;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        color: #000000;
    }
    
    p, span, div {
        color: #000000;
    }
    
    /* Hero Section */
    .hero {
        padding: 100px 5% 80px;
        background: linear-gradient(135deg, #00585A 0%, #00989B 100%);
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 3rem;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    
    .hero-content {
        max-width: 900px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    .hero h1 {
        font-size: 3.8rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        color: white;
    }
    
    .hero p {
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        opacity: 0.9;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        color: white;
    }
    
    .btn-primary {
        background: white;
        color: #00585A;
        padding: 0.9rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
        text-decoration: none;
    }
    
    .btn-primary:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 255, 255, 0.3);
        color: #00585A;
    }
    
    /* Section Styling */
    .section {
        padding: 60px 5%;
    }
    
    .section-title {
        text-align: center;
        margin-bottom: 4rem;
    }
    
    .section-title h2 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        position: relative;
        display: inline-block;
        color: #000000;
    }
    
    .section-title h2::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, #00585A 0%, #00989B 100%);
        border-radius: 2px;
    }
    
    .section-title p {
        font-size: 1.2rem;
        color: #000000;
        max-width: 700px;
        margin: 0 auto;
        opacity: 0.8;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2.5rem;
        margin-top: 3rem;
    }
    
    .feature-card {
        background: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #DCDEDD;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #00585A 0%, #00989B 100%);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0, 88, 90, 0.1);
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #00585A 0%, #00989B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        padding: 1rem;
        border-radius: 50%;
        background-color: rgba(0, 152, 155, 0.1);
    }
    
    .feature-card h4 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
        color: #000000;
    }
    
    .feature-card p {
        color: #000000;
        line-height: 1.7;
        opacity: 0.8;
    }
    
    /* Stats Section with Background Color */
    .stats-section {
        background: linear-gradient(135deg, #00585A 0%, #00989B 100%);
        padding: 80px 5%;
        text-align: center;
        color: white;
    }
    
    .stats-section .section-title h2 {
        color: white;
    }
    
    .stats-section .section-title p {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2.5rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
    }
    
    /* Data Preprocessing Section with Background */
    .data-preprocessing-section {
        background: linear-gradient(135deg, #DCDEDD 0%, #99A2A1 100%);
        padding: 80px 5%;
        color: #000000;
    }
    
    .data-preprocessing-section .section-title h2 {
        color: #000000;
    }
    
    .data-preprocessing-section .section-title p {
        color: #000000;
        opacity: 0.8;
    }
    
    /* Streamlit Overrides */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
    }
    
    .stButton button {
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #DCDEDD;
    }
    
    .stFileUploader {
        border: 2px dashed #DCDEDD;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #00989B;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 12px 12px 0 0;
        gap: 1rem;
        padding: 0 1.5rem;
        font-weight: 600;
        background: #F8FAFC;
        border: 1px solid #DCDEDD;
        color: #000000;
    }
    
    .stTabs [aria-selected="true"] {
        background: #00989B;
        color: white;
        border-color: #00989B;
    }
    
    /* Custom Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        background: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #DCDEDD;
        color: #000000;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #00585A 0%, #00181A 100%);
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1lcbmhc h1, .css-1lcbmhc h2, .css-1lcbmhc h3 {
        color: white !important;
    }
    
    /* Make all text black in main content */
    .stApp {
        color: #000000;
    }
    
    .stMarkdown {
        color: #000000;
    }
    
    .stAlert {
        color: #000000;
    }
    
    .stDataFrame {
        color: #000000;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

<!-- Hero Section -->
<div class="hero" id="home">
    <div class="hero-content">
        <h1>Predict MSME Performance with AI</h1>
        <p>Advanced machine learning platform for district-level MSME growth analysis using real registration data and predictive modeling.</p>
        <a href="#predict" class="btn-primary">Get Started</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
MODEL_DIR = r"E:\ML proj\Models"
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "district_model.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "district_features.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        return model, features, scaler
    except:
        st.error("Model not found. Please train the model first.")
        return None, None, None

model, features, scaler = load_model()

# Fixed function to handle Activities column
def safe_extract_activities(text):
    """
    Safely extract activities from the Activities column handling NaN and malformed data
    """
    if pd.isna(text) or text == 'nan' or text == '':
        return ""
    
    try:
        # Try to parse as JSON/literal eval first
        activities_list = ast.literal_eval(str(text))
        if isinstance(activities_list, list):
            return " ".join([str(a.get('Description', '')) for a in activities_list if isinstance(a, dict)])
        else:
            return str(text)
    except (ValueError, SyntaxError):
        # If literal eval fails, try JSON parsing
        try:
            activities_list = json.loads(str(text))
            if isinstance(activities_list, list):
                return " ".join([str(a.get('Description', '')) for a in activities_list if isinstance(a, dict)])
            else:
                return str(text)
        except (json.JSONDecodeError, TypeError):
            # If all parsing fails, return the string as is
            return str(text)

# Sidebar Navigation with enhanced styling
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-logo {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #99A2A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #00585A;
    }
    .nav-section {
        margin-bottom: 2rem;
    }
    .nav-section-title {
        color: #DCDEDD !important;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        padding-left: 1rem;
    }
    </style>
    
    <div class="sidebar-logo">📊 MSME Predictor</div>
    """, unsafe_allow_html=True)
    
    # Navigation sections
    st.markdown('<div class="nav-section-title" style="color: #DCDEDD !important;">MAIN NAVIGATION</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigate to:",
        ["Home", "Data Preprocessing", "Visualizations", "Predict"],
        index=0,
        key="main_nav"
    )
    
    st.markdown("---")
    
    # Quick Actions section
    st.markdown('<div class="nav-section-title" style="color: #DCDEDD !important;">QUICK ACTIONS</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Prediction", use_container_width=True):
        page = "Predict"
    
    if st.button("📊 View Visualizations", use_container_width=True):
        page = "Visualizations"
    
    if st.button("🔄 Process Data", use_container_width=True):
        page = "Data Preprocessing"
    
    st.markdown("---")
    
    # Model Status
    st.markdown('<div class="nav-section-title" style="color: #DCDEDD !important;">MODEL STATUS</div>', unsafe_allow_html=True)
    
    if model is not None:
        st.success("✅ Model Loaded")
        st.metric("Features", len(features))
    else:
        st.error("❌ Model Not Found")

# === HOME ===
if page == "Home":
    st.markdown("""
    <div class="section">
        <div class="section-title">
            <h2 style="color: #000000;">Key Features</h2>
            <p style="color: #000000;">Our platform provides comprehensive tools for MSME data analysis and performance prediction</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h4 style="color: #000000;">Advanced Analytics</h4>
                <p style="color: #000000;">Explore MSME density, micro ratios, and growth patterns with interactive visualizations and detailed insights.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h4 style="color: #000000;">SVM Prediction</h4>
                <p style="color: #000000;">High-accuracy classification using Support Vector Machines to categorize districts as Low, Medium, or High performance.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <h4 style="color: #000000;">Rich Visualizations</h4>
                <p style="color: #000000;">Interactive word clouds, heatmaps, PCA analysis, and performance dashboards for comprehensive data exploration.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section with colored background
    st.markdown("""
    <div class="stats-section">
        <div class="section-title">
            <h2 style="color: white;">Platform Statistics</h2>
            <p style="color: rgba(255, 255, 255, 0.9);">Our system processes and analyzes thousands of MSME records</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">15K+</div>
                <div class="stat-label">MSME Records</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">95%</div>
                <div class="stat-label">Prediction Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">36</div>
                <div class="stat-label">Districts Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Platform Availability</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# === DATA PREPROCESSING ===
elif page == "Data Preprocessing":
    st.markdown("""
    <div class="data-preprocessing-section">
        <div class="section-title">
            <h2 style="color: #000000;">Data Preprocessing</h2>
            <p style="color: #000000;">Clean and prepare your MSME data for analysis and modeling</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load and display data
    try:
        df = pd.read_csv(r"E:\ML proj\Data\msme_MAHARASHTRA.csv")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Columns", df.shape[1])
        
        with col2:
            st.dataframe(df.head(10), use_container_width=True)
        
        # Data processing
        with st.expander("Data Processing Steps", expanded=True):
            st.subheader("Text Processing")
            
            # Use the safe function to extract activities
            df['ActivityText'] = df['Activities'].apply(safe_extract_activities)
            df['AllText'] = df['EnterpriseName'].fillna("") + " " + df['ActivityText']

            def clean_text(t):
                words = re.findall(r'\b[a-zA-Z]{3,}\b', str(t).lower())
                stop = {'the','and','for','with','this','that','from','are','has','have'}
                return " ".join([w for w in words if w not in stop])
            
            df['CleanText'] = df['AllText'].apply(clean_text)

            st.write("**Cleaned Text Sample:**")
            st.dataframe(df[['EnterpriseName', 'CleanText']].head(8), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if the data file exists at the specified path")
    
    st.markdown("</div>", unsafe_allow_html=True)

# === VISUALIZATIONS ===
elif page == "Visualizations":
    st.markdown('<div class="section" id="visualize"><div class="section-title"><h2 style="color: #000000;">Visual Insights</h2><p style="color: #000000;">Interactive visualizations to explore MSME data patterns</p></div>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv(r"E:\ML proj\Data\msme_MAHARASHTRA.csv")
        
        # Use the safe function to extract activities
        df['ActivityText'] = df['Activities'].apply(safe_extract_activities)
        df['CleanText'] = df['EnterpriseName'].fillna("") + " " + df['ActivityText']
        df['CleanText'] = df['CleanText'].apply(lambda x: " ".join(re.findall(r'\b[a-zA-Z]{3,}\b', str(x).lower())))

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Word Cloud", "Business Distribution", "Data Overview"])
        
        with tab1:
            st.subheader("MSME Activities Word Cloud")
            all_text = " ".join(df['CleanText'])
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='viridis', 
                max_words=100,
                contour_color='#00585A',
                contour_width=1
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Common MSME Activities', fontsize=16, fontweight='bold', color='#000000')
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Business Types Distribution")
            counts = Counter(all_text.split())
            words = ['manufacture', 'retail', 'wholesale', 'service', 'food', 'textile', 'construction', 'transport']
            vals = [counts.get(w, 0) for w in words]
            
            # Create a more visually appealing pie chart with new colors
            colors = ['#00989B', '#00585A', '#99A2A1', '#DCDEDD', '#00181A', '#00777A', '#00888B', '#00666A']
            fig = px.pie(
                values=vals, 
                names=words, 
                color_discrete_sequence=colors,
                hole=0.4
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='white', width=2))
            )
            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Distribution of Business Types",
                title_x=0.5,
                font=dict(color='#000000')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Data Overview", anchor="data-overview")
            
            # Create a sample visualization of district data
            if 'district_name' in df.columns:
                district_counts = df['district_name'].value_counts().head(10)
                fig = px.bar(
                    x=district_counts.values, 
                    y=district_counts.index,
                    orientation='h',
                    title="Top 10 Districts by MSME Count",
                    color=district_counts.values,
                    color_continuous_scale=['#DCDEDD', '#00989B', '#00585A']
                )
                fig.update_layout(
                    xaxis_title="Number of MSMEs",
                    yaxis_title="District",
                    height=500,
                    font=dict(color='#000000'),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No district data available for visualization")
                
    except Exception as e:
        st.error(f"Error loading data for visualizations: {str(e)}")
        st.info("Please check if the data file exists and is properly formatted")

# === PREDICTION ===
elif page == "Predict":
    st.markdown('<div class="section" id="predict"><div class="section-title"><h2 style="color: #000000;">Predict Performance</h2><p style="color: #000000;">Upload your data and get AI-powered predictions for MSME performance</p></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model is not loaded. Please check if the model files exist.")
    else:
        uploaded = st.file_uploader("Upload test_data.csv", type="csv", 
                                   help="Upload a CSV file with district data for prediction")
        
        if uploaded and model:
            with st.spinner("Processing your data..."):
                try:
                    df = pd.read_csv(uploaded)
                    
                    # Create features
                    df['MSME_Density'] = df['total'] / 1000
                    df['Micro_Ratio'] = df['micro'] / df['total'].replace(0, 1)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    # Make predictions
                    X = df[features].fillna(0)
                    X_scaled = scaler.transform(X)
                    df['Predicted_Performance'] = model.predict(X_scaled)
                    df['Performance_Probability'] = model.predict_proba(X_scaled)[:, 2]
                    df['Performance_Label'] = df['Predicted_Performance'].map({0:'Low',1:'Medium',2:'High'})

                    st.success("✅ Prediction Complete!")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Prediction Results")
                        st.dataframe(df[['district_name','Performance_Label','Performance_Probability']], 
                                   use_container_width=True)
                    
                    with col2:
                        st.subheader("Summary")
                        low_count = (df['Performance_Label'] == 'Low').sum()
                        medium_count = (df['Performance_Label'] == 'Medium').sum()
                        high_count = (df['Performance_Label'] == 'High').sum()
                        
                        st.metric("Low Performance", low_count)
                        st.metric("Medium Performance", medium_count)
                        st.metric("High Performance", high_count)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name="msme_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualizations
                    st.subheader("Performance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Performance distribution
                        fig = px.pie(
                            df, 
                            names='Performance_Label', 
                            title="Performance Distribution",
                            color='Performance_Label',
                            color_discrete_map={
                                'Low':'#99A2A1', 
                                'Medium':'#00989B', 
                                'High':'#00585A'
                            }
                        )
                        fig.update_traces(
                            textposition='inside', 
                            textinfo='percent+label',
                            marker=dict(line=dict(color='white', width=2))
                        )
                        fig.update_layout(
                            font=dict(color='#000000')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Top performers
                        top = df[df['Predicted_Performance']==2].nlargest(10, 'Performance_Probability')
                        if not top.empty:
                            fig = px.bar(
                                top, 
                                y='district_name', 
                                x='Performance_Probability',
                                title="Top 10 High Performers",
                                orientation='h',
                                color='Performance_Probability',
                                color_continuous_scale=['#DCDEDD', '#00989B', '#00585A']
                            )
                            fig.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                font=dict(color='#000000')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No high performers in this dataset")
                            
                except Exception as e:
                    st.error(f"Error processing prediction data: {str(e)}")

# Simple footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #000000; border-top: 1px solid #DCDEDD; margin-top: 3rem;">
    <p style="color: #000000;">© 2025 MSME Performance Predictor</p>
</div>
""", unsafe_allow_html=True)