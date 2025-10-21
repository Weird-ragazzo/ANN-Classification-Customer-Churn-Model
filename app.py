import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    .subtitle {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# Header
st.title('📊 Customer Churn Prediction')
st.markdown('<p class="subtitle">Predict customer churn probability using advanced machine learning</p>', unsafe_allow_html=True)

st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Customer Information")
    
    # Create tabs for different input categories
    tab1, tab2, tab3 = st.tabs(["📋 Personal Details", "💰 Financial Info", "🔧 Account Details"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
            gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        with col_b:
            age = st.slider('🎂 Age', 18, 92, 35)
            tenure = st.slider('📅 Tenure (years)', 0, 10, 5)
    
    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650, step=1)
            balance = st.number_input('💵 Account Balance', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        with col_d:
            estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    
    with tab3:
        col_e, col_f = st.columns(2)
        with col_e:
            num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
            has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        with col_f:
            is_active_member = st.selectbox('✅ Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col2:
    st.header("ℹ️ Quick Tips")
    # Use Streamlit native info box to ensure visibility across Streamlit versions
    st.info(
        """
        About Churn Prediction

        This model analyzes customer data to predict the likelihood of churn. Key factors include:
        - Account activity status
        - Number of products
        - Age and tenure
        - Financial indicators
        """
    )

st.markdown("---")

# Predict button
if st.button('🚀 Predict Churn'):
    with st.spinner('Analyzing customer data...'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display results
        st.header("📈 Prediction Results")
        
        # Create metric card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-value">{prediction_proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Result interpretation
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if prediction_proba > 0.5:
                st.markdown(f"""
                <div class="warning-card">
                    <h3>⚠️ High Churn Risk</h3>
                    <p>The customer is <strong>likely to churn</strong> with a probability of <strong>{prediction_proba:.1%}</strong></p>
                    <p>Consider implementing retention strategies.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card">
                    <h3>✅ Low Churn Risk</h3>
                    <p>The customer is <strong>not likely to churn</strong> with a probability of <strong>{prediction_proba:.1%}</strong></p>
                    <p>Customer appears satisfied with services.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            # Risk level gauge
            risk_level = "High" if prediction_proba > 0.7 else "Medium" if prediction_proba > 0.3 else "Low"
            st.metric("Risk Level", risk_level, delta=f"{prediction_proba:.1%}")
            
            # Confidence
            confidence = abs(prediction_proba - 0.5) * 2
            st.metric("Prediction Confidence", f"{confidence:.1%}")
        
        # Recommendations
        if prediction_proba > 0.5:
            st.subheader("💡 Recommended Actions")
            st.markdown("""
            - 🎯 Reach out to customer with personalized offers
            - 📞 Schedule a follow-up call to address concerns
            - 🎁 Consider loyalty rewards or discounts
            - 📧 Send targeted retention campaigns
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Powered by TensorFlow & Streamlit | Built with ❤️ for better customer retention</p>
</div>
""", unsafe_allow_html=True)