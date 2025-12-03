import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from sklearn.manifold import TSNE
import plotly.express as px

st.set_page_config(layout="wide", page_title="CardioCare — Heart Risk & Clustering")


@st.cache_data
def load_data(proj_dir):
    X_train = pd.read_csv(os.path.join(proj_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(proj_dir, "y_train.csv"))
    return X_train, y_train

def cap_bp(X):
    X = X.copy()
    X['ap_hi'] = X['ap_hi'].clip(0, 250)
    X['ap_lo'] = X['ap_lo'].clip(0, 200)
    return X

def preprocess_fit(X):
    Xc = cap_bp(X)
    scaler = RobustScaler()
    num_cols = ['age', 'height', 'weight']

    scaler.fit(Xc[num_cols])
    km = KMeans(n_clusters=6, random_state=42)
    km.fit(scaler.transform(Xc[num_cols]))
    return scaler, km

@st.cache_resource
def load_xgb(proj_dir):
    model_path = os.path.join(proj_dir, "xgb_model_modified.json")
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

# App UI

st.title("CardioCare — Heart Disease Risk Prediction & Patient Clustering")

proj_dir = os.path.dirname(__file__)
X_train, y_train = load_data(proj_dir)

# Fit scaler & clustering
scaler, km = preprocess_fit(X_train)
bst = load_xgb(proj_dir)

# Input UI
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=0)
gender_str = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_str == "Female" else 2

height = st.sidebar.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=0.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=0.0)

ap_hi = st.sidebar.number_input("Systolic BP", min_value=0, max_value=300, value=0)
ap_lo = st.sidebar.number_input("Diastolic BP", min_value=0, max_value=250, value=0)

cholesterol_str = st.sidebar.selectbox(
    "Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
cholesterol = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[cholesterol_str]

gluc_str = st.sidebar.selectbox(
    "Glucose", ["Normal", "Above Normal", "Well Above Normal"])
gluc = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[gluc_str]

smoke = 1 if st.sidebar.selectbox("Smoking", ["No", "Yes"]) == "Yes" else 0
alco  = 1 if st.sidebar.selectbox("Alcohol Use", ["No", "Yes"]) == "Yes" else 0
active= 1 if st.sidebar.selectbox("Physical Activity", ["No", "Yes"]) == "Yes" else 0

submit = st.sidebar.button("Predict")

# Prediction Logic

if submit:
    # Create df for the single patient
    df_input = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }])

    # Preprocess
    df_clean = cap_bp(df_input)
    X_proc = df_clean.copy()
    X_proc[['age','height','weight']] = scaler.transform(df_clean[['age','height','weight']])

    # Match training columns
    X_for_model = X_proc[X_train.columns]

    dmat = xgb.DMatrix(X_for_model)
    pred_prob = float(bst.predict(dmat)[0])

    # Output
    st.subheader("Prediction Result")
    st.metric("Heart Disease Risk Probability", f"{pred_prob:.3f}")
    st.write("Risk Level:", "**HIGH**" if pred_prob >= 0.5 else "**LOW**")

    # Clustering based on numeric data
    cluster = km.predict(scaler.transform(df_clean[['age','height','weight']]))[0]
    st.write(f"Assigned Patient Cluster: **{cluster}**")

    # Nearest Similar Patients
    st.subheader("Most Similar Profiles")
    train_scaled = cap_bp(X_train.copy())
    train_scaled[['age','height','weight']] = scaler.transform(train_scaled[['age','height','weight']])

    nn = NearestNeighbors(n_neighbors=5).fit(train_scaled[['age','height','weight']])
    dists, idxs = nn.kneighbors(X_proc[['age','height','weight']])

    st.dataframe(X_train.iloc[idxs[0]])

    # 2D Visualization
    st.subheader("Patient Clusters Overview (t-SNE)")
    sample = X_train.sample(1500, random_state=42)
    sample_scaled = cap_bp(sample.copy())
    sample_scaled[['age','height','weight']] = scaler.transform(sample_scaled[['age','height','weight']])

    tsne = TSNE(n_components=2, random_state=42)
    proj = tsne.fit_transform(sample_scaled[['age','height','weight']])
    sample['tsne1'], sample['tsne2'] = proj[:,0], proj[:,1]
    sample['cluster'] = km.predict(sample_scaled[['age','height','weight']])

    fig = px.scatter(
        sample,
        x="tsne1", y="tsne2",
        color="cluster",
        hover_data=["age", "gender", "ap_hi", "ap_lo"]
    )
    st.plotly_chart(fig, use_container_width=True)

