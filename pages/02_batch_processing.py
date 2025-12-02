import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.batch_processor import BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Cannot import BatchProcessor: {e}")
    BATCH_PROCESSOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Batch Processing - Cardiovascular Risk",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
    .download-btn {
        background-color: #10B981 !important;
    }
    .download-btn:hover {
        background-color: #059669 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">📊 Batch Cardiovascular Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>⚡ Process multiple patients at once</strong><br>
    Upload a CSV file with patient data to assess cardiovascular risk for entire populations.
    The system will preprocess data, run predictions, and assign risk clusters.
</div>
""", unsafe_allow_html=True)

# Initialize batch processor
@st.cache_resource
def get_batch_processor():
    if BATCH_PROCESSOR_AVAILABLE:
        return BatchProcessor()
    return None

processor = get_batch_processor()

if not BATCH_PROCESSOR_AVAILABLE:
    st.error("""
    ❌ Batch processor module not found. Please ensure:
    1. `utils/batch_processor.py` exists
    2. It contains the `BatchProcessor` class
    3. Required dependencies are installed
    """)
    st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📋 Data Guide", "ℹ️ About"])

with tab1:
    st.markdown('<h2 class="sub-header">1. Upload Patient Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with patient data. Click 'Download Template' for the correct format.",
            key="file_uploader"
        )
    
    with col2:
        st.markdown("### Get Template")
        if st.button("📥 Download CSV Template", use_container_width=True):
            try:
                template_csv = processor.get_csv_template()
                st.download_button(
                    label="Click to Download",
                    data=template_csv,
                    file_name="cardiovascular_template.csv",
                    mime="text/csv",
                    key="download_template"
                )
            except Exception as e:
                st.error(f"Error generating template: {e}")
    
    # Process uploaded file
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">2. Process Data</h2>', unsafe_allow_html=True)
        
        if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
            with st.spinner("Processing your data..."):
                processed_df, error = processor.process_batch(uploaded_file)
            
            if error:
                st.error(f"Processing failed: {error}")
            elif processed_df is not None:
                # Display success message
                st.markdown(f'<div class="success-box"><strong>✅ Processing Complete!</strong><br>Successfully analyzed {len(processed_df)} patients.</div>', unsafe_allow_html=True)
                
                # Store processed data in session state
                st.session_state.processed_data = processed_df
                st.session_state.original_filename = uploaded_file.name
                
                # Display summary statistics
                st.markdown('<h2 class="sub-header">3. Results Summary</h2>', unsafe_allow_html=True)
                
                summary = processor.generate_summary_statistics(processed_df)
                
                # Metrics in cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6B7280;">Total Patients</div>
                        <div style="font-size: 2rem; font-weight: bold; color: #1F2937;">{summary['total_patients']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6B7280;">High Risk Patients</div>
                        <div style="font-size: 2rem; font-weight: bold; color: #DC2626;">{summary['high_risk_count']}</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">{summary['high_risk_percentage']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6B7280;">Avg Risk Score</div>
                        <div style="font-size: 2rem; font-weight: bold; color: #1F2937;">{summary['avg_risk_score']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    cluster_count = len(summary['cluster_distribution'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6B7280;">Clusters Identified</div>
                        <div style="font-size: 2rem; font-weight: bold; color: #1F2937;">{cluster_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown('<h2 class="sub-header">4. Visualizations</h2>', unsafe_allow_html=True)
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Risk distribution pie chart
                    high_risk = summary['high_risk_count']
                    low_risk = summary['total_patients'] - high_risk
                    
                    fig1 = px.pie(
                        values=[high_risk, low_risk],
                        names=['High Risk', 'Low Risk'],
                        title="Risk Distribution",
                        color_discrete_sequence=['#EF4444', '#10B981'],
                        hole=0.4
                    )
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with viz_col2:
                    # Cluster distribution bar chart
                    if summary['cluster_distribution']:
                        clusters = list(summary['cluster_distribution'].keys())
                        counts = list(summary['cluster_distribution'].values())
                        
                        fig2 = px.bar(
                            x=clusters,
                            y=counts,
                            title="Patient Distribution by Cluster",
                            labels={'x': 'Cluster ID', 'y': 'Number of Patients'},
                            color=counts,
                            color_continuous_scale='Blues'
                        )
                        fig2.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed results table
                st.markdown('<h2 class="sub-header">5. Detailed Results</h2>', unsafe_allow_html=True)
                
                # Show important columns only
                display_columns = [
                    'age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
                    'prediction_probability', 'risk_category', 'cluster', 'processing_status'
                ]
                
                # Filter to columns that exist
                existing_columns = [col for col in display_columns if col in processed_df.columns]
                
                if 'bmi' in processed_df.columns:
                    existing_columns.append('bmi')
                
                # Add original filename column
                processed_df_display = processed_df[existing_columns].copy()
                processed_df_display.insert(0, 'patient_id', range(1, len(processed_df_display) + 1))
                
                st.dataframe(
                    processed_df_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download section
                st.markdown('<h2 class="sub-header">6. Download Results</h2>', unsafe_allow_html=True)
                
                download_col1, download_col2, download_col3 = st.columns([1, 1, 2])
                
                with download_col1:
                    output_format = st.selectbox(
                        "Format",
                        ['csv', 'excel'],
                        index=0,
                        key="output_format"
                    )
                
                with download_col2:
                    include_all_cols = st.checkbox("Include all columns", value=True)
                
                with download_col3:
                    if st.button("💾 Download Full Results", type="secondary", use_container_width=True, key="download_full"):
                        try:
                            # Prepare data for download
                            if include_all_cols:
                                download_df = processed_df
                            else:
                                download_df = processed_df_display
                            
                            filename = processor.save_results(download_df, output_format)
                            
                            # Read file and create download button
                            with open(filename, 'rb') as f:
                                file_data = f.read()
                            
                            st.download_button(
                                label=f"📥 Download {output_format.upper()} File",
                                data=file_data,
                                file_name=filename,
                                mime="text/csv" if output_format == 'csv' else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="download_results"
                            )
                            
                            # Clean up temporary file
                            import time
                            time.sleep(2)
                            if os.path.exists(filename):
                                os.remove(filename)
                                
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                
                # Data quality report
                with st.expander("🔍 Data Quality Report", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Missing values
                        missing_data = processed_df.isnull().sum()
                        missing_total = missing_data.sum()
                        
                        if missing_total > 0:
                            st.warning(f"⚠️ Found {missing_total} missing values")
                            st.write("Missing values per column:")
                            st.dataframe(missing_data[missing_data > 0])
                        else:
                            st.success("✅ No missing values detected")
                    
                    with col2:
                        # Processing status
                        completed = summary['completed_rows']
                        errors = summary['error_rows']
                        
                        if errors > 0:
                            st.warning(f"⚠️ {errors} rows had processing errors")
                            error_rows = processed_df[processed_df['processing_status'].str.contains('Error', na=False)]
                            st.dataframe(error_rows[['processing_status']].head())
                        else:
                            st.success(f"✅ All {completed} rows processed successfully")

with tab2:
    st.markdown('<h2 class="sub-header">📋 Data Format Guide</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Your CSV file must contain these 11 columns with the exact names and formats:
    </div>
    """, unsafe_allow_html=True)
    
    # Column specifications table
    column_specs = pd.DataFrame({
        'Column': ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                  'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
        'Description': [
            'Age in years (or days, will auto-convert)',
            '1 = female, 2 = male',
            'Height in centimeters (cm)',
            'Weight in kilograms (kg)',
            'Systolic blood pressure (mmHg)',
            'Diastolic blood pressure (mmHg)',
            '1 = normal, 2 = above normal, 3 = well above normal',
            '1 = normal, 2 = above normal, 3 = well above normal',
            '0 = non-smoker, 1 = smoker',
            '0 = non-drinker, 1 = drinker',
            '0 = not active, 1 = active'
        ],
        'Example': [
            '45 (years) or 16425 (days)',
            '1 or 2',
            '175',
            '80',
            '120',
            '80',
            '1, 2, or 3',
            '1, 2, or 3',
            '0 or 1',
            '0 or 1',
            '0 or 1'
        ]
    })
    
    st.dataframe(column_specs, use_container_width=True, hide_index=True)
    
    # Sample data
    st.markdown("### Sample Data Preview")
    sample_data = {
        'age': [45, 52, 38],
        'gender': [1, 2, 1],
        'height': [175, 162, 180],
        'weight': [80, 65, 90],
        'ap_hi': [120, 140, 130],
        'ap_lo': [80, 90, 85],
        'cholesterol': [1, 2, 3],
        'gluc': [1, 1, 2],
        'smoke': [0, 1, 0],
        'alco': [0, 1, 0],
        'active': [1, 1, 0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    # Preprocessing information
    st.markdown("### ⚙️ What Happens During Processing")
    
    preprocessing_steps = [
        "🔹 **Age Conversion**: Days to years (if age > 100)",
        "🔹 **BP Correction**: Fixes swapped systolic/diastolic values",
        "🔹 **Outlier Removal**: Removes extreme values using IQR method",
        "🔹 **Encoding**: Converts categorical values to numbers",
        "🔹 **Scaling**: Normalizes numerical features",
        "🔹 **BMI Calculation**: Computes Body Mass Index",
        "🔹 **Prediction**: Assesses cardiovascular risk",
        "🔹 **Clustering**: Groups patients by similar profiles"
    ]
    
    for step in preprocessing_steps:
        st.markdown(f"<div style='margin: 0.5rem 0;'>{step}</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<h2 class="sub-header">ℹ️ About This Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Cardiovascular Risk Batch Assessment Tool</strong><br>
    This application uses machine learning models to assess cardiovascular disease risk 
    for multiple patients simultaneously.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Features")
    
    features = [
        "✅ **Batch Processing**: Upload and process hundreds of patients at once",
        "✅ **Automatic Preprocessing**: Cleans and prepares data automatically",
        "✅ **Risk Prediction**: Uses XGBoost model for accurate risk assessment",
        "✅ **Patient Clustering**: Groups similar patient profiles together",
        "✅ **Comprehensive Reports**: Detailed statistics and visualizations",
        "✅ **Export Results**: Download processed data in CSV or Excel format"
    ]
    
    for feature in features:
        st.markdown(f"<div style='margin: 0.5rem 0;'>{feature}</div>", unsafe_allow_html=True)
    
    st.markdown("### 🔧 Technical Details")
    
    tech_details = [
        "**Backend**: Python with scikit-learn and XGBoost",
        "**Frontend**: Streamlit web framework",
        "**Preprocessing**: RobustScaler, LabelEncoder, IQR outlier removal",
        "**Model**: XGBoost classifier trained on cardiovascular data",
        "**Output**: Risk probability, confidence intervals, cluster assignments"
    ]
    
    for detail in tech_details:
        st.markdown(f"<div style='margin: 0.5rem 0;'>{detail}</div>", unsafe_allow_html=True)
    
    st.markdown("### 📞 Support")
    st.markdown("""
    For issues or questions:
    - Check the data format requirements
    - Ensure all required columns are present
    - Verify data types are correct
    - Contact your system administrator for technical support
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;'>
    Cardiovascular Risk Assessment System • For medical research purposes only
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None

# Add refresh button in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    if st.button("🔄 Clear Session & Start New", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Processing Info")
    
    if st.session_state.processed_data is not None:
        st.success(f"✅ Data loaded: {len(st.session_state.processed_data)} patients")
        if st.button("📄 View Current Results", use_container_width=True):
            st.dataframe(st.session_state.processed_data.head())
    else:
        st.info("No data processed yet")
    
    st.markdown("---")
    st.markdown("### 🔗 Navigation")
    st.page_link("app.py", label="🏠 Single Patient Assessment", icon="🏠")
    st.page_link("pages/02_Batch_Processing.py", label="📊 Batch Processing", icon="📊", disabled=True)

# Run test if this file is executed directly
if __name__ == "__main__":
    st.info("This is the Batch Processing page. Run the main app with `streamlit run app.py`")