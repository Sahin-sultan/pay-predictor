import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pay Predict - AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for orange theme (changing from blue to orange)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    
    .header-container {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
        border: 1px solid rgba(255, 107, 53, 0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #ffffff;
        margin-top: 10px;
        opacity: 0.9;
    }
    
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4); }
        to { box-shadow: 0 4px 25px rgba(255, 107, 53, 0.8); }
    }
    
    .social-links {
        margin-top: 20px;
    }
    
    .social-links a {
        color: #ffffff;
        text-decoration: none;
        margin: 0 10px;
        font-size: 1.1rem;
        transition: color 0.3s;
    }
    
    .social-links a:hover {
        color: #ffcc00;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #ff6b35;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .result-amount {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b35;
        text-align: center;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stSelectbox > div > div {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    
    .stSlider > div > div > div {
        color: #ff6b35;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        return model, label_encoders
    except:
        st.error("⚠️ Model files not found. Please run train_model.py first!")
        st.stop()

# Load data for exploration
@st.cache_data
def load_data():
    return pd.read_csv('indian_salary_data_500.csv')

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">💰 Pay Predict</h1>
        <p class="header-subtitle">AI-Powered Indian Salary Prediction System</p>
        <div class="accuracy-badge">🎯 89% Accuracy</div>
        <div class="social-links">
            <a href="https://github.com/Sahin-sultan" target="_blank">📁 GitHub</a>
            <a href="https://linkedin.com/in/sahin-sultan-917b50331" target="_blank">💼 LinkedIn</a>
            <a href="https://www.instagram.com/sahin_edition/" target="_blank">📸 Instagram</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, label_encoders = load_model()
    df = load_data()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Prediction Controls")
    st.sidebar.markdown("---")
    
    # Create two columns for main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📊 Salary Prediction")
        
        # Input form
        with st.form("prediction_form"):
            # Age input
            age = st.slider(
                "👤 Age",
                min_value=22,
                max_value=60,
                value=30,
                help="Select your age"
            )
            
            # Gender selection
            gender = st.selectbox(
                "⚥ Gender",
                options=['Male', 'Female'],
                help="Select your gender"
            )
            
            # Education selection
            education = st.selectbox(
                "🎓 Education Level",
                options=sorted(df['education'].unique()),
                help="Select your highest education level"
            )
            
            # Years of experience
            experience = st.slider(
                "💼 Years of Experience",
                min_value=0,
                max_value=30,
                value=5,
                help="Select your total years of professional experience"
            )
            
            # Job title
            job_title = st.selectbox(
                "👨‍💻 Job Title",
                options=sorted(df['job_title'].unique()),
                help="Select your job role"
            )
            
            # Job location type
            job_location = st.selectbox(
                "🏢 Location Type",
                options=sorted(df['job_location'].unique()),
                help="Select your work location type"
            )
            
            # City
            city = st.selectbox(
                "🏙️ City",
                options=sorted(df['city'].unique()),
                help="Select your work city"
            )
            
            # Nationality
            nationality = st.selectbox(
                "🌍 Nationality",
                options=sorted(df['nationality'].unique()),
                help="Select your nationality"
            )
            
            # Predict button
            submitted = st.form_submit_button("🔮 Predict Salary", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = pd.DataFrame({
                    'age': [age],
                    'gender': [gender],
                    'education': [education],
                    'years_of_experience': [experience],
                    'job_title': [job_title],
                    'job_location': [job_location],
                    'city': [city],
                    'nationality': [nationality]
                })
                
                # Encode categorical variables
                for column in ['gender', 'education', 'job_title', 'job_location', 'city', 'nationality']:
                    if column in label_encoders:
                        try:
                            input_data[column] = label_encoders[column].transform(input_data[column])
                        except:
                            # Handle unseen categories
                            input_data[column] = 0
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #ff6b35; text-align: center;">💰 Predicted Annual Salary</h3>
                    <div class="result-amount">₹{prediction:.1f} Lakhs</div>
                    <p style="text-align: center; color: #cccccc;">
                        Based on Indian market data and AI analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📈 Salary Insights")
                
                # Compare with similar profiles
                similar_profiles = df[
                    (df['job_title'] == job_title) & 
                    (df['city'] == city) &
                    (abs(df['years_of_experience'] - experience) <= 2)
                ]
                
                if len(similar_profiles) > 0:
                    avg_salary = similar_profiles['salary_inr_lakhs'].mean()
                    min_salary = similar_profiles['salary_inr_lakhs'].min()
                    max_salary = similar_profiles['salary_inr_lakhs'].max()
                    
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        st.metric("📊 Market Average", f"₹{avg_salary:.1f}L")
                    
                    with col_insight2:
                        st.metric("📉 Market Range Min", f"₹{min_salary:.1f}L")
                    
                    with col_insight3:
                        st.metric("📈 Market Range Max", f"₹{max_salary:.1f}L")
    
    with col2:
        st.markdown("## 📊 Data Exploration")
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["💼 Job Analysis", "🏙️ City Analysis", "📚 Education Analysis"])
        
        with tab1:
            # Salary by job title
            job_salary = df.groupby('job_title')['salary_inr_lakhs'].agg(['mean', 'count']).reset_index()
            job_salary = job_salary[job_salary['count'] >= 5].sort_values('mean', ascending=True)
            
            fig = px.bar(
                job_salary, 
                x='mean', 
                y='job_title',
                title='Average Salary by Job Title',
                color='mean',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Salary by city
            city_salary = df.groupby('city')['salary_inr_lakhs'].agg(['mean', 'count']).reset_index()
            city_salary = city_salary[city_salary['count'] >= 10].sort_values('mean', ascending=True)
            
            fig = px.bar(
                city_salary, 
                x='mean', 
                y='city',
                title='Average Salary by City',
                color='mean',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Salary by education
            edu_salary = df.groupby('education')['salary_inr_lakhs'].mean().sort_values(ascending=True)
            
            fig = px.bar(
                x=edu_salary.values,
                y=edu_salary.index,
                title='Average Salary by Education Level',
                color=edu_salary.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics section
    st.markdown("---")
    st.markdown("## 📈 Dataset Statistics")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("📊 Total Records", len(df))
    
    with col_stat2:
        st.metric("💰 Avg Salary", f"₹{df['salary_inr_lakhs'].mean():.1f}L")
    
    with col_stat3:
        st.metric("🏢 Cities Covered", df['city'].nunique())
    
    with col_stat4:
        st.metric("👨‍💻 Job Roles", df['job_title'].nunique())

if __name__ == "__main__":
    main()
    # Bottom navbar
    st.markdown("""
        <div style=\"position: fixed; left: 0; bottom: 0; width: 100%; background: #1e1e1e; color: #cccccc; text-align: center; padding: 12px 0; font-size: 1rem; z-index: 100;\">
            © 2025 Pay Predict. All rights reserved.<br>
            Developed by <a href=\"https://github.com/Sahin-sultan\" style=\"color: #4fc3f7; text-decoration: none; font-weight: bold;\" target=\"_blank\">Sahin Sultan</a>
        </div>
    """, unsafe_allow_html=True)
