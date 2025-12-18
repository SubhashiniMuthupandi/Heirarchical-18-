import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- SETTINGS ---
FOLDER_NAME = "Model10"
FILE_NAME = "hierarchical_model.pkl"

# Path logic
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, FOLDER_NAME, FILE_NAME)

st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("📂 Hierarchical Clustering Predictor")
st.write("This app uses the Hierarchical Clustering model to group applicants.")

# --- LOAD MODEL ---
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    st.sidebar.success("Model loaded from Model10")

    # --- INPUT FIELDS ---
    st.subheader("Enter Applicant Details:")
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
    
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    # --- PREDICTION ---
    if st.button("Determine Cluster"):
        # Prepare input data
        input_data = np.array([[income, credit_score, loan_amount]])
        
        try:
            # Note: Standard AgglomerativeClustering doesn't have .predict()
            # If your model supports it, this will work:
            prediction = model.fit_predict(input_data) 
            cluster_id = prediction[0]
            
            st.write("---")
            st.header(f"Result: Applicant belongs to Cluster {cluster_id}")
            st.info("Clusters represent groups of people with similar financial profiles.")
            
        except AttributeError:
            st.error("Error: This specific Hierarchical model type does not support predicting new individual data points.")
            st.warning("Hierarchical clustering is usually used to analyze a whole dataset at once, not one-by-one predictions.")

else:
    st.error(f"❌ File Not Found!")
    st.info(f"Please ensure your file is at: `{model_path}`")
