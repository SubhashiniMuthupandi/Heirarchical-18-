import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# --- SETTINGS ---
FOLDER_NAME = "Model10"
FILE_NAME = "hierarchical_model.pkl"
DATA_FILE = "1loan_approval.csv"

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, FOLDER_NAME, FILE_NAME)
data_path = os.path.join(base_path, DATA_FILE)

st.set_page_config(page_title="Applicant Group Finder", layout="wide")
st.title("📂 Applicant Segmentation & Grouping")

# --- LOAD MODEL ---
if os.path.exists(model_path) and os.path.exists(data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data for profiling
    df = pd.read_csv(data_path)
    features = ["Income", "Credit Score", "Loan Amount"]
    
    st.sidebar.info("Model and Data loaded successfully.")

    # --- INPUT ---
    st.write("### Enter Applicant Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        income = st.number_input("Annual Income ($)", value=60000)
    with col2:
        credit = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    with col3:
        loan = st.number_input("Loan Amount ($)", value=20000)

    if st.button("Find My Group"):
        # 1. Combine user data with original data
        user_data = pd.DataFrame([[income, credit, loan]], columns=features)
        combined_df = pd.concat([df[features], user_data], ignore_index=True)

        # 2. Fit model to get clusters
        clusters = model.fit_predict(combined_df)
        user_cluster = clusters[-1] # The user's cluster

        # 3. PROFILE THE GROUPS
        # We calculate the average of each cluster to understand what they represent
        combined_df['Cluster'] = clusters
        analysis = combined_df.groupby('Cluster')[features].mean()

        # 4. DEFINE GROUP NAMES (Logic based on averages)
        # We create a dictionary to describe each group
        group_descriptions = {}
        for i in range(len(analysis)):
            avg_income = analysis.loc[i, 'Income']
            avg_credit = analysis.loc[i, 'Credit Score']
            
            if avg_income > df['Income'].mean() and avg_credit > 650:
                group_descriptions[i] = "🌟 Premium Group (High Income, High Credit)"
            elif avg_income < df['Income'].mean() and avg_credit < 600:
                group_descriptions[i] = "⚠️ High Risk Group (Low Income, Low Credit)"
            elif avg_income > df['Income'].mean() and avg_credit < 600:
                group_descriptions[i] = "💰 Middle Group (High Income, Low Credit)"
            else:
                group_descriptions[i] = "📊 Standard Group (Average Financial Profile)"

        # --- DISPLAY RESULTS ---
        st.write("---")
        st.header(f"You belong to: {group_descriptions[user_cluster]}")
        
        # Show a comparison table
        st.write("### How you compare to the group averages:")
        comparison_df = analysis.loc[[user_cluster]].copy()
        comparison_df.index = ["Group Average"]
        user_row = pd.DataFrame([[income, credit, loan]], columns=features, index=["You"])
        st.table(pd.concat([user_row, comparison_df]))

else:
    if not os.path.exists(model_path):
        st.error(f"Missing Model File: {model_path}")
    if not os.path.exists(data_path):
        st.error(f"Missing Data File: {data_path}. Please upload '1loan_approval.csv' to the main folder.")
