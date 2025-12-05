import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from sklearn.manifold import TSNE
import plotly.express as px
import shap

# -------------------------
# App config & logger
# -------------------------
st.set_page_config(layout="wide", page_title="CardioCare — Heart Risk & Clustering")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Helpers & safe utilities
# -------------------------
@st.cache_data
def load_csv_safely(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(proj_dir):
    x_path = os.path.join(proj_dir, "X_train.csv")
    y_path = os.path.join(proj_dir, "y_train.csv")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Required training CSVs not found in project directory")
    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)
    return X_train, y_train

# Cap blood pressure to sensible clinical bounds
def cap_bp(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if 'ap_hi' in X.columns:
        X['ap_hi'] = X['ap_hi'].clip(80, 250)  # realistic systolic lower bound 80
    if 'ap_lo' in X.columns:
        X['ap_lo'] = X['ap_lo'].clip(40, 200)  # realistic diastolic lower bound 40
    return X

# Fit preprocessing + KMeans on numeric features
@st.cache_resource
def preprocess_fit(X: pd.DataFrame, num_cols=None, n_clusters=6):
    if num_cols is None:
        num_cols = ['age', 'height', 'weight']
    Xc = cap_bp(X)
    scaler = RobustScaler()
    scaler.fit(Xc[num_cols])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(scaler.transform(Xc[num_cols]))
    return scaler, km

@st.cache_resource
def load_xgb(proj_dir):
    model_path = os.path.join(proj_dir, "xgb_model_modified.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"xgboost model file not found at: {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

# small helper to validate single patient inputs
def validate_single_input(d):
    errors = []
    if d['age'] <= 0 or d['age'] > 120:
        errors.append('Age must be between 1 and 120')
    if d['height'] <= 50 or d['height'] > 250:
        errors.append('Height seems unrealistic (must be between 51 and 250 cm)')
    if d['weight'] <= 10 or d['weight'] > 500:
        errors.append('Weight seems unrealistic (must be between 11 and 500 kg)')
    if d['ap_hi'] <= 50 or d['ap_hi'] > 300:
        errors.append('Systolic BP out of range')
    if d['ap_lo'] <= 30 or d['ap_lo'] > 250:
        errors.append('Diastolic BP out of range')
    return errors

# Display shap via matplotlib fallback with safe guards
def plot_shap_force(explainer, shap_values, X_row):
    try:
        # shap.force_plot with matplotlib rendering
        fig = shap.plots._force_matplotlib.force_matplotlib(explainer.expected_value, shap_values, X_row)
        st.pyplot(fig)
    except Exception as e:
        logger.warning(f"Matplotlib SHAP force failed: {e}")
        # fallback: bar chart (feature contributions)
        try:
            contrib = pd.Series(shap_values, index=X_row.index)
            contrib_df = contrib.sort_values()
            st.bar_chart(contrib_df)
        except Exception as e2:
            logger.error(f"Fallback SHAP plotting also failed: {e2}")
            st.write("Unable to render SHAP plot for this prediction.")

# -------------------------
# App UI & logic
# -------------------------
st.title("CardioCare — Heart Disease Risk Prediction & Patient Clustering")

proj_dir = os.path.dirname(__file__)

# Load baseline training data (used for schema/columns and nearest-neighbors)
try:
    X_train, y_train = load_data(proj_dir)
except Exception as e:
    st.error(f"Error loading training data: {e}")
    st.stop()

# Precompute scaler and clustering
scaler, km = preprocess_fit(X_train)

# Load xgboost model
try:
    bst = load_xgb(proj_dir)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Patient Input Mode")
input_mode = st.sidebar.radio("Select Input Mode", ["Single Patient", "Batch Upload"])

if input_mode == "Single Patient":
    st.sidebar.header("Patient Information")
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=30)
    gender_str = st.sidebar.selectbox("Gender", ["Female", "Male"])  # keep original encoding
    gender = 1 if gender_str == "Female" else 2
    height = st.sidebar.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
    weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0)
    ap_hi = st.sidebar.number_input("Systolic BP", min_value=0, max_value=300, value=120)
    ap_lo = st.sidebar.number_input("Diastolic BP", min_value=0, max_value=250, value=80)
    cholesterol_str = st.sidebar.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    cholesterol = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[cholesterol_str]
    gluc_str = st.sidebar.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = {"Normal":1, "Above Normal":2, "Well Above Normal":3}[gluc_str]
    smoke = 1 if st.sidebar.selectbox("Smoking", ["No", "Yes"]) == "Yes" else 0
    alco  = 1 if st.sidebar.selectbox("Alcohol Use", ["No", "Yes"]) == "Yes" else 0
    active= 1 if st.sidebar.selectbox("Physical Activity", ["No", "Yes"]) == "Yes" else 0

    bootstrap_iters = st.sidebar.slider("Bootstrap iterations (CI estimate)", min_value=10, max_value=500, value=50, step=10, help="Fewer iterations = faster response; more = tighter CI")
    submit = st.sidebar.button("Predict Single Patient")

    if submit:
        df_input = pd.DataFrame([{
            'age': age, 'gender': gender, 'height': height, 'weight': weight,
            'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
            'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active
        }])

        # Validate
        val_errs = validate_single_input(df_input.iloc[0].to_dict())
        if val_errs:
            for e in val_errs:
                st.warning(e)
            st.stop()

        # Clean
        df_clean = cap_bp(df_input)
        X_proc = df_clean.copy()
        num_cols = ['age','height','weight']
        X_proc[num_cols] = scaler.transform(df_clean[num_cols])

        # Ensure columns match training set
        missing_cols = [c for c in X_train.columns if c not in X_proc.columns]
        if missing_cols:
            st.error(f"Model expects columns missing from input: {missing_cols}")
            st.stop()

        X_for_model = X_proc[X_train.columns]

        # Prediction
        try:
            dmat = xgb.DMatrix(X_for_model)
            pred_prob = float(bst.predict(dmat)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Bootstrap CI (sampling with replacement from the same single row is a naive CI — we keep it but note it's approximate)
        probs_boot = []
        for _ in range(bootstrap_iters):
            dmat_boot = xgb.DMatrix(X_for_model.sample(frac=1.0, replace=True))
            probs_boot.append(float(bst.predict(dmat_boot).mean()))
        ci_low, ci_high = np.percentile(probs_boot, [2.5, 97.5])

        st.subheader("Prediction Result")
        st.metric("Heart Disease Risk Probability", f"{pred_prob:.3f}")
        st.write(f"Risk Level: **{'HIGH' if pred_prob >= 0.5 else 'LOW'}**")
        st.write(f"Confidence Interval (approx. 95%): [{ci_low:.3f}, {ci_high:.3f}]")

        # Clustering
        cluster = int(km.predict(scaler.transform(df_clean[['age','height','weight']]))[0])
        st.write(f"Assigned Patient Cluster: **{cluster}**")

        # Nearest Similar Patients
        st.subheader("Most Similar Profiles")
        train_scaled = cap_bp(X_train.copy())
        train_scaled[['age','height','weight']] = scaler.transform(train_scaled[['age','height','weight']])
        nn = NearestNeighbors(n_neighbors=5).fit(train_scaled[['age','height','weight']])
        dists, idxs = nn.kneighbors(X_proc[['age','height','weight']])
        st.dataframe(X_train.iloc[idxs[0]])

        # SHAP explanation
        st.subheader("Feature Contribution (SHAP)")
        try:
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(X_for_model)
            # shap_values may be 2D (n_samples x n_features) or a list for multiclass; handle common case
            if isinstance(shap_values, list):
                sv = shap_values[0]
            else:
                sv = shap_values
            plot_shap_force(explainer, sv[0], X_for_model.iloc[0])
        except Exception as e:
            logger.error(f"SHAP generation failed: {e}")
            st.write("SHAP explanation unavailable for this model / prediction.")

else:
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload patient CSV", type=["csv"]) 
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        required_cols = set(X_train.columns)
        if not required_cols.issubset(set(batch_df.columns)):
            missing = required_cols.difference(set(batch_df.columns))
            st.error(f"Uploaded CSV is missing required columns: {sorted(list(missing))}")
            st.stop()

        st.subheader("Batch Predictions")
        df_clean = cap_bp(batch_df.copy())
        num_cols = ['age','height','weight']
        df_clean[num_cols] = scaler.transform(df_clean[num_cols])
        X_for_model = df_clean[X_train.columns]

        try:
            dmat = xgb.DMatrix(X_for_model)
            pred_probs = bst.predict(dmat)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.stop()

        batch_df['Heart_Disease_Risk'] = pred_probs
        batch_df['Risk_Level'] = np.where(pred_probs>=0.5, "HIGH", "LOW")
        batch_df['Cluster'] = km.predict(df_clean[['age','height','weight']])
        st.dataframe(batch_df)
        csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv_bytes, file_name="batch_predictions.csv")

# -----------------------------------
# Cluster Overview (visual)
# -----------------------------------
try:
    st.subheader("Patient Clusters Overview (t-SNE)")
    sample = X_train.sample(min(1500, len(X_train)), random_state=42)
    sample_scaled = cap_bp(sample.copy())
    sample_scaled[['age','height','weight']] = scaler.transform(sample_scaled[['age','height','weight']])
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
    proj = tsne.fit_transform(sample_scaled[['age','height','weight']])
    sample['tsne1'], sample['tsne2'] = proj[:,0], proj[:,1]
    sample['cluster'] = km.predict(sample_scaled[['age','height','weight']])
    fig = px.scatter(
        sample, x="tsne1", y="tsne2", color=sample['cluster'].astype(str),
        hover_data=["age", "gender", "ap_hi", "ap_lo"],
        title="t-SNE projection of patient clusters"
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    logger.error(f"Cluster overview failed: {e}")
    st.write("Cluster overview unavailable.")

# footer
st.markdown("---")
st.write("App ready — data privacy note: ensure PHI is not uploaded to public deployments without appropriate protections.")
