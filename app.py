import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- SETTINGS ---
FOLDER_NAME = "Model10"
FILE_NAME = "hierarchical_model.pkl"

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, FOLDER_NAME, FILE_NAME)

st.title("📂 Hierarchical Clustering Predictor")

# --- LOAD MODEL ---
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    st.success("Model Loaded")

    # User Input
    income = st.number_input("Annual Income", min_value=0, value=67000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)

    if st.button("Determine Cluster"):
        # 1. Create the new data point
        new_data = np.array([[income, credit_score, loan_amount]])

        # 2. FIX: Hierarchical clustering needs at least 2 points.
        # We will create a "dummy" point (or you could load your CSV here) 
        # so the model has enough data to run.
        dummy_data = np.array([[50000, 600, 15000]]) # A standard reference point
        
        # Combine them so there are 2 rows
        combined_data = np.vstack([dummy_data, new_data])

        try:
            # 3. Run fit_predict on the combined data
            clusters = model.fit_predict(combined_data)
            
            # The result for the user's input is the last item in the array
            user_cluster = clusters[-1]
            
            st.write("---")
            st.header(f"Result: Applicant assigned to Cluster {user_cluster}")
            st.info("Note: Clustering groups similar people together based on income and credit.")

        except Exception as e:
            st.error(f"Cluster Error: {e}")
else:
    st.error(f"Model file not found at {model_path}")
