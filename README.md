# Customer Churn Prediction App 📊

A beautiful and interactive Streamlit application that predicts customer churn probability using machine learning.

## 🌐 Live Demo

**Try it now:** [https://anncustomer-churn-model.streamlit.app](https://anncustomer-churn-model.streamlit.app)

## Features ✨

- 🎨 Modern, gradient-based UI design
- 📊 Real-time churn probability predictions
- 🔍 Organized input fields with tabs
- 📈 Visual risk assessment and confidence metrics
- 💡 Actionable recommendations for high-risk customers

## Requirements 📋

```
streamlit
tensorflow
scikit-learn
pandas
numpy
```

## Installation 🚀

1. Clone the repository or download the files

2. Install dependencies:
```bash
pip install streamlit tensorflow scikit-learn pandas numpy
```

3. Ensure you have the following files in your project directory:
   - `model.h5` - Trained TensorFlow model
   - `label_encoder_gender.pkl` - Gender label encoder
   - `onehot_encoder_geo.pkl` - Geography one-hot encoder
   - `scaler.pkl` - Feature scaler

## Usage 💻

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## How to Use 🎯

1. **Enter Customer Information:**
   - Personal Details: Geography, Gender, Age, Tenure
   - Financial Info: Credit Score, Balance, Salary
   - Account Details: Products, Credit Card, Active Status

2. **Click "Predict Churn"** to get the prediction

3. **View Results:**
   - Churn probability percentage
   - Risk level assessment
   - Prediction confidence
   - Recommended actions (if high risk)

## Model Information 🤖

The app uses a trained neural network model that analyzes customer features to predict churn probability. Key factors include:

- Credit score and account balance
- Customer demographics (age, geography)
- Account activity and tenure
- Number of products and services used

## Output Interpretation 📖

- **Probability > 50%**: Customer is likely to churn (High Risk)
- **Probability ≤ 50%**: Customer is not likely to churn (Low Risk)

The app also provides:
- Risk Level: Low, Medium, or High
- Prediction Confidence: How certain the model is
- Actionable recommendations for retention

## License 📄

This project is open source and available under the MIT License.

## Support 💬

For issues or questions, please open an issue in the repository.

---

Built with ❤️ using Streamlit and TensorFlow
