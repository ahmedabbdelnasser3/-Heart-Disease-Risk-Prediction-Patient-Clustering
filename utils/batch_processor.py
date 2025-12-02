import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import sys
import os
from sklearn.preprocessing import RobustScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class BatchProcessor:
    """
    Batch processor for cardiovascular risk assessment
    Handles CSV uploads, processes multiple patients, generates predictions
    """
    
    def __init__(self):
        # Define required columns for input CSV
        self.required_columns = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active'
        ]
        
        # Column descriptions for users
        self.column_descriptions = {
            'age': 'Age (in years or days)',
            'gender': '1 = female, 2 = male',
            'height': 'Height in cm',
            'weight': 'Weight in kg',
            'ap_hi': 'Systolic blood pressure',
            'ap_lo': 'Diastolic blood pressure',
            'cholesterol': '1 = normal, 2 = above normal, 3 = well above normal',
            'gluc': '1 = normal, 2 = above normal, 3 = well above normal',
            'smoke': '0 = non-smoker, 1 = smoker',
            'alco': '0 = non-drinker, 1 = drinker',
            'active': '0 = not active, 1 = active'
        }
        
        # Initialize preprocessing objects (like your notebook)
        self.scaler = RobustScaler()
        self.le_cholesterol = LabelEncoder()
        self.le_gluc = LabelEncoder()
        
        # Fit encoders with expected values (like your notebook)
        self.le_cholesterol.fit([1, 2, 3])  # Your data has 1,2,3
        self.le_gluc.fit([1, 2, 3])         # Your data has 1,2,3
        
        # Load your trained model
        self.model = self.load_model()
        
    def load_model(self):
        """Load your trained XGBoost model"""
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model('xgb_model_modified.json')  
            print("✅ XGBoost model loaded successfully")
            return model
        except Exception as e:
            print(f"⚠️ Could not load XGBoost model: {e}")
            print("⚠️ Using rule-based fallback")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess data EXACTLY like your Graduation project.ipynb notebook
        """
        df_processed = df.copy()
        
        # 1. AGE CONVERSION: days to years 
        if 'age' in df_processed.columns:
            # If average age > 100, it's probably in days
            if df_processed['age'].mean() > 100:
                # Convert days to years 
                df_processed['age'] = (df_processed['age'] / 365).round(1)
        
        # 2. FIX SWAPPED BP VALUES
        mask = df_processed['ap_lo'] > df_processed['ap_hi']
        if mask.any():
            df_processed.loc[mask, ['ap_hi', 'ap_lo']] = df_processed.loc[mask, ['ap_lo', 'ap_hi']].values
        
        # 3. REMOVE EXTREME BP VALUES 
        df_processed = df_processed[(df_processed['ap_hi'] <= 250) & (df_processed['ap_lo'] <= 200)]
        
        # 4. REMOVE OUTLIERS USING IQR METHOD 
        
        # For ap_hi (systolic BP)
        Q1_hi = df_processed['ap_hi'].quantile(0.25)
        Q3_hi = df_processed['ap_hi'].quantile(0.75)
        IQR_hi = Q3_hi - Q1_hi
        lower_hi = Q1_hi - 1.5 * IQR_hi
        upper_hi = Q3_hi + 1.5 * IQR_hi
        df_processed = df_processed[(df_processed['ap_hi'] >= lower_hi) & (df_processed['ap_hi'] <= upper_hi)]
        
        # For ap_lo (diastolic BP)
        Q1_lo = df_processed['ap_lo'].quantile(0.25)
        Q3_lo = df_processed['ap_lo'].quantile(0.75)
        IQR_lo = Q3_lo - Q1_lo
        lower_lo = Q1_lo - 1.5 * IQR_lo
        upper_lo = Q3_lo + 1.5 * IQR_lo
        df_processed = df_processed[(df_processed['ap_lo'] >= lower_lo) & (df_processed['ap_lo'] <= upper_lo)]
        
        # For height 
        Q1_height = df_processed['height'].quantile(0.25)
        Q3_height = df_processed['height'].quantile(0.75)
        IQR_height = Q3_height - Q1_height
        lower_height = Q1_height - 1.5 * IQR_height
        upper_height = Q3_height + 1.5 * IQR_height
        df_processed = df_processed[(df_processed['height'] >= lower_height) & (df_processed['height'] <= upper_height)]
        
        # For weight 
        Q1_weight = df_processed['weight'].quantile(0.25)
        Q3_weight = df_processed['weight'].quantile(0.75)
        IQR_weight = Q3_weight - Q1_weight
        lower_weight = Q1_weight - 1.5 * IQR_weight
        upper_weight = Q3_weight + 1.5 * IQR_weight
        df_processed = df_processed[(df_processed['weight'] >= lower_weight) & (df_processed['weight'] <= upper_weight)]
        
        # For age 
        if 'age' in df_processed.columns:
            Q1_age = df_processed['age'].quantile(0.25)
            Q3_age = df_processed['age'].quantile(0.75)
            IQR_age = Q3_age - Q1_age
            lower_age = Q1_age - 1.5 * IQR_age
            upper_age = Q3_age + 1.5 * IQR_age
            df_processed = df_processed[(df_processed['age'] >= lower_age) & (df_processed['age'] <= upper_age)]
        
        # 5. CALCULATE BMI 
        if 'height' in df_processed.columns and 'weight' in df_processed.columns:
            df_processed['bmi'] = df_processed['weight'] / ((df_processed['height']/100) ** 2)
        
        # 6. ENCODE CATEGORICAL FEATURES 
        if 'cholesterol' in df_processed.columns:
            try:
                df_processed['cholesterol'] = self.le_cholesterol.transform(df_processed['cholesterol'].astype(int))
            except:
                # If new values, map them (1→0, 2→1, 3→2)
                df_processed['cholesterol'] = df_processed['cholesterol'].apply(
                    lambda x: 0 if x == 1 else (1 if x == 2 else 2)
                )
        
        if 'gluc' in df_processed.columns:
            try:
                df_processed['gluc'] = self.le_gluc.transform(df_processed['gluc'].astype(int))
            except:
                df_processed['gluc'] = df_processed['gluc'].apply(
                    lambda x: 0 if x == 1 else (1 if x == 2 else 2)
                )
        
        # 7. SCALE NUMERICAL FEATURES
        numerical_cols = ['age', 'height', 'weight']
        existing_numerical = [col for col in numerical_cols if col in df_processed.columns]
        
        if existing_numerical:
            if hasattr(self.scaler, 'scale_'): 
                df_processed[existing_numerical] = self.scaler.transform(df_processed[existing_numerical])
            else:
                # Initial fit with reasonable values
                dummy_data = pd.DataFrame({
                    'age': [30, 40, 50, 60, 70],
                    'height': [150, 160, 170, 180, 190],
                    'weight': [50, 60, 70, 80, 90]
                })
                self.scaler.fit(dummy_data[existing_numerical])
                df_processed[existing_numerical] = self.scaler.transform(df_processed[existing_numerical])
        
        # 8. ENSURE ALL COLUMNS ARE INTEGER TYPE 
        for col in ['gender', 'smoke', 'alco', 'active']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        return df_processed
    
    def predict_with_model(self, row):
        """
        Predict using your XGBoost model or fallback
        """
        if self.model:
            try:
                import xgboost as xgb
                
                # Prepare data for prediction
                row_df = pd.DataFrame([row])
                
                # Add BMI if not present
                if 'height' in row_df.columns and 'weight' in row_df.columns and 'bmi' not in row_df.columns:
                    row_df['bmi'] = row_df['weight'] / ((row_df['height']/100) ** 2)
                
                # Your XGBoost model expects these columns
                # Adjust based on your actual model training
                expected_columns = [
                    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
                ]
                
                # Add BMI if we calculated it
                if 'bmi' in row_df.columns:
                    expected_columns.append('bmi')
                
                # Keep only columns that exist
                available_cols = [col for col in expected_columns if col in row_df.columns]
                row_df = row_df[available_cols]
                
                # Convert to DMatrix (XGBoost format)
                dmatrix = xgb.DMatrix(row_df)
                
                # Predict
                prediction = self.model.predict(dmatrix)[0]
                return float(prediction)
                
            except Exception as e:
                st.warning(f"Model prediction failed: {e}. Using rule-based fallback.")
                return self.rule_based_prediction(row)
        else:
            return self.rule_based_prediction(row)
    
    def rule_based_prediction(self, patient_data):
        """
        Fallback prediction based on risk factors (if model fails)
        """
        risk_score = 0.0
        
        # Age factor (after conversion to years)
        age = patient_data.get('age', 50)
        if age > 60:
            risk_score += 0.4
        elif age > 50:
            risk_score += 0.3
        elif age > 40:
            risk_score += 0.2
        
        # Blood pressure factor
        ap_hi = patient_data.get('ap_hi', 120)
        if ap_hi > 160:
            risk_score += 0.5
        elif ap_hi > 140:
            risk_score += 0.3
        elif ap_hi > 130:
            risk_score += 0.2
        
        # Cholesterol factor
        cholesterol = patient_data.get('cholesterol', 1)
        if cholesterol == 3:  # well above normal
            risk_score += 0.3
        elif cholesterol == 2:  # above normal
            risk_score += 0.15
        
        # Glucose factor
        gluc = patient_data.get('gluc', 1)
        if gluc == 3:
            risk_score += 0.2
        elif gluc == 2:
            risk_score += 0.1
        
        # Smoking factor
        if patient_data.get('smoke', 0) == 1:
            risk_score += 0.2
        
        # Alcohol factor
        if patient_data.get('alco', 0) == 1:
            risk_score += 0.1
        
        # Inactivity factor
        if patient_data.get('active', 1) == 0:
            risk_score += 0.1
        
        # BMI factor (if available)
        if 'bmi' in patient_data:
            bmi = patient_data['bmi']
            if bmi > 30:
                risk_score += 0.3
            elif bmi > 25:
                risk_score += 0.15
        
        # Normalize to 0-1 range
        prediction_prob = min(1.0, max(0.0, risk_score))
        
        return prediction_prob
    
    def validate_csv(self, df):
        """
        Validate uploaded CSV file has required columns and data types
        """
        # Check for missing columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        
        if missing_cols:
            error_msg = f"❌ Missing required columns: {missing_cols}\n\n"
            error_msg += "📋 Required columns and their format:\n"
            for col in self.required_columns:
                error_msg += f"- {col}: {self.column_descriptions[col]}\n"
            raise ValueError(error_msg)
        
        # Check for missing values
        missing_values = df[self.required_columns].isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"⚠️ Found {missing_values} missing values. They will be handled during preprocessing.")
        
        # Convert numeric columns
        numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def process_batch(self, uploaded_file):
        """
        Main function to process batch CSV file
        """
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            st.info(f"📁 File uploaded: {uploaded_file.name}")
            st.write(f"📊 Original data: {len(df)} rows, {len(df.columns)} columns")
            
            # Validate CSV structure
            df = self.validate_csv(df)
            
            # Preprocess data 
            st.write("🔄 Preprocessing data (cleaning, scaling, encoding)...")
            df_processed = self.preprocess_data(df)
            
            if len(df_processed) == 0:
                st.error("❌ No valid data after preprocessing. Check for extreme BP values or outliers.")
                return None, "No valid data after preprocessing"
            
            st.write(f"📈 After preprocessing: {len(df_processed)} valid rows")
            
            # Initialize result columns
            df_processed['prediction'] = 0
            df_processed['prediction_probability'] = 0.0
            df_processed['confidence_interval_lower'] = 0.0
            df_processed['confidence_interval_upper'] = 0.0
            df_processed['cluster'] = 0
            df_processed['risk_category'] = 'Low'
            df_processed['processing_status'] = 'Pending'
            df_processed['bmi'] = 0.0  # Initialize BMI column
            
            # Calculate BMI for all rows
            if 'height' in df_processed.columns and 'weight' in df_processed.columns:
                df_processed['bmi'] = df_processed['weight'] / ((df_processed['height']/100) ** 2)
            
            # Process each row
            total_rows = len(df_processed)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, row in df_processed.iterrows():
                try:
                    # Update progress
                    progress = (idx + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"🔍 Processing patient {idx + 1} of {total_rows}")
                    
                    # PREDICT using your model or rule-based
                    prediction_prob = self.predict_with_model(row)
                    
                    # Calculate confidence interval (±10%)
                    ci_lower = max(0.0, prediction_prob - 0.1)
                    ci_upper = min(1.0, prediction_prob + 0.1)
                    
                    # Assign cluster (placeholder - Member 4 will implement)
                    cluster_id = self.assign_cluster(row)
                    
                    # Update DataFrame with results
                    df_processed.at[idx, 'prediction'] = 1 if prediction_prob > 0.5 else 0
                    df_processed.at[idx, 'prediction_probability'] = round(prediction_prob, 4)
                    df_processed.at[idx, 'confidence_interval_lower'] = round(ci_lower, 4)
                    df_processed.at[idx, 'confidence_interval_upper'] = round(ci_upper, 4)
                    df_processed.at[idx, 'cluster'] = cluster_id
                    df_processed.at[idx, 'risk_category'] = 'High' if prediction_prob > 0.5 else 'Low'
                    df_processed.at[idx, 'processing_status'] = 'Completed'
                    
                except Exception as row_error:
                    df_processed.at[idx, 'processing_status'] = f'Error: {str(row_error)[:50]}...'
                    continue
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Add original data back for reference (keeping original values)
            for col in df.columns:
                if col not in df_processed.columns:
                    df_processed[col + '_original'] = df[col]
            
            st.success(f"✅ Successfully processed {len(df_processed)} patients!")
            
            # Show processing statistics
            completed = len(df_processed[df_processed['processing_status'] == 'Completed'])
            errors = len(df_processed) - completed
            
            if errors > 0:
                st.warning(f"⚠️ {errors} rows had errors during processing")
            
            return df_processed, None
            
        except Exception as e:
            error_msg = f"❌ Batch processing failed: {str(e)}"
            st.error(error_msg)
            return None, error_msg
    
    def assign_cluster(self, row):
        """
        Placeholder for cluster assignment
        Member 4 will implement actual clustering
        """
        # Simple rule-based clustering for now
        age = row.get('age', 50)
        
        # Calculate BMI
        if 'height' in row and 'weight' in row:
            bmi = row['weight'] / ((row['height']/100) ** 2)
        else:
            bmi = 25
        
        # Cluster logic based on age and BMI
        if age > 60:
            if bmi > 30:
                return 0  # Older, obese
            elif bmi > 25:
                return 1  # Older, overweight
            else:
                return 2  # Older, normal weight
        else:
            if bmi > 30:
                return 3  # Younger, obese
            elif bmi > 25:
                return 4  # Younger, overweight
            else:
                return 5  # Younger, normal weight
    
    def generate_summary_statistics(self, df):
        """
        Generate summary statistics from processed data
        """
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_patients': len(df),
            'high_risk_count': int(df['prediction'].sum()),
            'high_risk_percentage': round((df['prediction'].sum() / len(df)) * 100, 2),
            'cluster_distribution': df['cluster'].value_counts().sort_index().to_dict(),
            'avg_risk_score': round(df['prediction_probability'].mean(), 4),
            'min_risk_score': round(df['prediction_probability'].min(), 4),
            'max_risk_score': round(df['prediction_probability'].max(), 4),
            'completed_rows': len(df[df['processing_status'] == 'Completed']),
            'error_rows': len(df[df['processing_status'].str.contains('Error', na=False)]),
            'avg_age': round(df['age'].mean(), 1) if 'age' in df.columns else 0,
            'avg_bmi': round(df['bmi'].mean(), 1) if 'bmi' in df.columns else 0
        }
        
        return summary
    
    def save_results(self, df, output_format='csv'):
        """
        Save processed results to file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'csv':
            filename = f"batch_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif output_format == 'excel':
            filename = f"batch_results_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
        else:
            filename = f"batch_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
        
        return filename
    
    def get_csv_template(self):
        """
        Create and return a CSV template for users
        """
        template_data = {
            'age': [45, 52, 38],  # Age in YEARS
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
        
        template_df = pd.DataFrame(template_data)
        
        # Add explanation row as comment
        explanation = "# Required columns with example values\n"
        explanation += "# age: in years (or days, will auto-convert)\n"
        explanation += "# gender: 1=female, 2=male\n"
        explanation += "# height: in cm\n"
        explanation += "# weight: in kg\n"
        explanation += "# ap_hi: systolic BP (90-250)\n"
        explanation += "# ap_lo: diastolic BP (60-200)\n"
        explanation += "# cholesterol: 1=normal, 2=above normal, 3=well above normal\n"
        explanation += "# gluc: 1=normal, 2=above normal, 3=well above normal\n"
        explanation += "# smoke: 0=non-smoker, 1=smoker\n"
        explanation += "# alco: 0=non-drinker, 1=drinker\n"
        explanation += "# active: 0=not active, 1=active\n"
        
        csv_content = explanation + template_df.to_csv(index=False)
        return csv_content


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("✅ Batch Processor Module Loaded Successfully")
    print("=" * 60)
    
    processor = BatchProcessor()
    
    print("📋 Required columns:")
    for col in processor.required_columns:
        print(f"  - {col}: {processor.column_descriptions[col]}")
    
    print(f"\n🔧 Model status: {'Loaded' if processor.model else 'Not loaded (using fallback)'}")
    print("=" * 60)
    
    # Test template generation
    template = processor.get_csv_template()
    print("\n📝 Sample template generated successfully")
    print("First 3 lines of template:")
    print(template.split('\n')[0:4])