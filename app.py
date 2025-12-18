import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import kagglehub

# --- SETTINGS ---
FOLDER_NAME = "Model10"
FILE_NAME = "hierarchical_model.pkl"
DATA_FILENAME = "1loan_approval.csv"

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, FOLDER_NAME, FILE_NAME)

st.set_page_config(page_title="Applicant Group Finder", layout="wide")
st.title("📂 Applicant Segmentation & Grouping")

# --- DATA LOADING LOGIC ---
@st.cache_data
def get_data():
    local_path = os.path.join(base_path, DATA_FILENAME)
    
    # Check if file is in the current folder
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    
    # If not, download from Kaggle (like in your notebook)
    try:
        st.info("Downloading reference dataset from Kaggle...")
        download_path = kagglehub.dataset_download("nmnbabbar/loan-approval")
        # Find the csv file in the downloaded path
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    return pd.read_csv(full_path)
    except Exception as e:
        st.error(f"Could not download data: {e}")
        return None

# Load the data
df = get_data()

# --- LOAD MODEL ---
if os.path.exists(model_path) and df is not None:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    features = ["Income", "Credit Score", "Loan Amount"]
    
    # --- UI INPUT ---
    st.write("### Enter Applicant Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        income = st.number_input("Annual Income ($)", value=60000)
    with col2:
        credit = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    with col3:
        loan = st.number_input("Loan Amount ($)", value=20000)

    if st.button("Find My Group"):
        # Combine user data with original data
        user_data = pd.DataFrame([[income, credit, loan]], columns=features)
        # We use a copy of the features from the original data
        df_clean = df[features].dropna()
        combined_df = pd.concat([df_clean, user_data], ignore_index=True)

        # Fit model
        clusters = model.fit_predict(combined_df)
        user_cluster = clusters[-1]

        # Analyze Clusters
        combined_df['Cluster'] = clusters
        analysis = combined_df.groupby('Cluster')[features].mean()

        # Logic for Group Names
        group_names = {}
        for i in range(len(analysis)):
            avg_inc = analysis.loc[i, 'Income']
            avg_cred = analysis.loc[i, 'Credit Score']
            
            if avg_inc > df_clean['Income'].mean() and avg_cred > 600:
                group_names[i] = "🌟 Premium Group (High Income/Good Credit)"
            elif avg_inc < df_clean['Income'].mean() and avg_cred < 600:
                group_names[i] = "⚠️ High Risk Group (Low Income/Low Credit)"
            else:
                group_names[i] = "📊 Standard Group (Balanced Profile)"

        # --- DISPLAY RESULTS ---
        st.success(f"### Result: {group_names.get(user_cluster, 'General Group')}")
        
        st.write("#### Comparison to Group Average")
        user_row = pd.DataFrame([[income, credit, loan]], columns=features, index=["You"])
        avg_row = analysis.loc[[user_cluster]].copy()
        avg_row.index = ["Group Average"]
        st.table(pd.concat([user_row, avg_row]))

elif df is None:
    st.error("The dataset file '1loan_approval.csv' is missing and the auto-download failed.")
else:
    st.error(f"Model file not found at {model_path}. Please check your Model10 folder.")
