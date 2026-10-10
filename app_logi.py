import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")
st.title("Logistic Regression Model Predictor")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('logistic.sav')

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Check feature dimensions to build inputs dynamically
if hasattr(model, 'n_features_in_'):
    num_features = model.n_features_in_
    feature_names = getattr(model, 'feature_names_in_', [f"Feature {i+1}" for i in range(num_features)])
else:
    # Fallback if standard scikit-learn attribute is missing
    num_features = 1
    feature_names = ["Feature 1"]

st.write("### Input Features")
input_data = []
for name in feature_names:
    val = st.number_input(f"Enter value for {name}", value=0.0)
    input_data.append(val)

# Prediction section
if st.button("Predict"):
    features = np.array([input_data])
    prediction = model.predict(features)
    
    st.write("### Result")
    st.metric(label="Predicted Class", value=int(prediction[0]))
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        st.write("#### Class Probabilities")
        for idx, prob in enumerate(probabilities):
            st.write(f"Class {idx}: {prob * 100:.2f}%")
