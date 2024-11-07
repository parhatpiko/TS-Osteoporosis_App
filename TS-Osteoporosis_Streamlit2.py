import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# Load the logistic regression model
try:
    model = joblib.load('LR_Tuned_TS.pkl')
except FileNotFoundError:
    st.error("Model file 'LR_Tuned_TS.pkl' not found. Please ensure it is in the correct directory.")
    st.stop()

# Load the explainer object from the file
try:
    with open('explainer_TS.pkl', 'rb') as f:
        explainer = pickle.load(f)
except FileNotFoundError:
    st.error("Explainer file 'explainer_TS.pkl' not found. Please ensure it is in the correct directory.")
    st.stop()

# Define feature names
feature_names = [
    'Neutrophil', 'HB', 'ESR', 'TC', 'K', 'eGFR', 'CYS_C', 'TP', 'ALB', 'LDL'
]

# Streamlit user interface
st.title("Osteoporosis Risk Predictor for Spinal Tuberculosis Patients")

# Create three columns
col1, col2, col3 = st.columns([1, 1, 1])

# Left column: Input section
with col1:
    st.header("Indicators")
    # Define input fields for each feature
    inputs = {}
    for feature in feature_names:
        if feature == 'Neutrophil':
            inputs[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0, value='')
        elif feature == 'HB':
            inputs[feature] = st.number_input(f"{feature} (Hemoglobin):", min_value=0.0, max_value=200.0, value='')
        elif feature == 'ESR':
            inputs[feature] = st.number_input(f"{feature} (Erythrocyte Sedimentation Rate):", min_value=0.0, max_value=100.0, value='')
        elif feature == 'TC':
            inputs[feature] = st.number_input(f"{feature} (Total Cholesterol):", min_value=0.0, max_value=400.0, value='')
        elif feature == 'K':
            inputs[feature] = st.number_input(f"{feature} (Potassium):", min_value=0.0, max_value=10.0, value='')
        elif feature == 'eGFR':
            inputs[feature] = st.number_input(f"{feature} (Estimated Glomerular Filtration Rate):", min_value=0.0, max_value=120.0, value='')
        elif feature == 'CYS_C':
            inputs[feature] = st.number_input(f"{feature} (Cystatin C):", min_value=0.0, max_value=100.0, value='')
        elif feature == 'TP':
            inputs[feature] = st.number_input(f"{feature} (Total Protein):", min_value=0.0, max_value=100.0, value='')
        elif feature == 'ALB':
            inputs[feature] = st.number_input(f"{feature} (Albumin):", min_value=0.0, max_value=100.0, value='')
        elif feature == 'LDL':
            inputs[feature] = st.number_input(f"{feature} (Low-Density Lipoprotein):", min_value=0.0, max_value=200.0, value='')

# Process inputs and make predictions
feature_values = [inputs[feature] for feature in feature_names]
features_df = pd.DataFrame([feature_values], columns=feature_names)
features = np.array([feature_values])
class_dict = {1:'Positive Outcome', 0:'Negative Outcome'}
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {class_dict[predicted_class]}")
    st.write(f"**Prediction Probabilities:** {predicted_proba[predicted_class] * 100:.1f}%")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of osteoporosis. "
            f"The model predicts that your probability of having osteoporosis is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a healthcare professional for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of osteoporosis. "
            f"The model predicts that your probability of not having osteoporosis is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your bone health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)
    shap_values = explainer.shap_values(features_df)
    combined_shap_values = np.vstack((shap_values, -shap_values))
    combined_shap_expected_value = [explainer.expected_value, -explainer.expected_value]
    class_name = ['Positive Outcome', 'Negative Outcome']
    # Middle and right columns: Explanation for both classes
    for which_class, col in enumerate([col2, col3]):
        with col:
            st.header(f"{class_name[which_class]} Explanation")
            st.subheader(f"SHAP Force Plot")
            fig = plt.figure()
            shap.force_plot(
                base_value=combined_shap_expected_value[which_class],
                shap_values=combined_shap_values[which_class],
                features=features_df,
                feature_names=feature_names,
                matplotlib=True,
                text_rotation=30
            )
            fig.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

            st.subheader(f"SHAP Waterfall Plot")
            fig = plt.figure()
            shap.waterfall_plot(
                shap.Explanation(base_values=combined_shap_expected_value[which_class],
                                 values=combined_shap_values[which_class],
                                 data=feature_values,
                                 feature_names=feature_names)
            )
            fig.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
