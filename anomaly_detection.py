import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import traceback
import sys

# 1. LOAD DATA
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Remove commas from duration and convert to float
    if df['duration'].dtype == 'object':
        df['duration'] = df['duration'].str.replace(',', '').astype(float)
    
    # Fill missing values for destination_name
    df['destination_name'] = df['destination_name'].fillna('Unknown')
    return df

# 2. PREPROCESS DATA
def preprocess_features(df):
    # Select features for the model
    features = ['duration', 'charge', 'city', 'destination_type', 'call_direction']
    X = df[features].copy()
    
    # Label Encode categorical columns
    le_dict = {}
    for col in ['city', 'destination_type', 'call_direction']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le # Store encoders to use on new data later
        
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, le_dict

# 3. RUN ALGORITHMS
def run_anomaly_detection(X_scaled, df):
    # a. Isolation Forest (Best for global outliers)
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['iso_forest_anomaly'] = iso.fit_predict(X_scaled)
    
    # b. Local Outlier Factor (Best for local density outliers)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    df['lof_anomaly'] = lof.fit_predict(X_scaled)
    
    # c. One-Class SVM (Boundary based)
    oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
    df['oc_svm_anomaly'] = oc_svm.fit_predict(X_scaled)
    
    return df, iso

# 4. VISUALIZATION
def plot_results(X_scaled, df):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ['iso_forest_anomaly', 'lof_anomaly', 'oc_svm_anomaly']
    titles = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
    
    for i, col in enumerate(models):
        sns.scatterplot(data=df, x='pca1', y='pca2', hue=col, 
                        palette={1: 'blue', -1: 'red'}, ax=axes[i], alpha=0.5)
        axes[i].set_title(titles[i])
    
    plt.savefig('anomaly_comparison_local.png')
    print("Visualization saved as 'anomaly_comparison_local.png'")

# MAIN EXECUTION
if __name__ == "__main__":
    # Load
    print("Starting anomaly_detection script...")
    file_path = "C:/Users/HP/Downloads/ProjectRBSN/January_masked_sample.csv"  # Ensure file is in the same folder
    data = load_and_clean_data(file_path)
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
    
    # Sample for speed (optional)
    data_sample = data.sample(20000, random_state=42).copy()
    print(f"Using data sample with {len(data_sample)} rows")
    
    # Preprocess
    X_scaled, scaler, le_encoders = preprocess_features(data_sample)
    print("Preprocessing complete")
    
    # Detect
    try:
        processed_df, best_model = run_anomaly_detection(X_scaled, data_sample)
        print("Anomaly detection complete")
    except Exception as e:
        print("Error during anomaly detection:", e)
        traceback.print_exc()
        sys.exit(1)
    
    # Visualize
    try:
        plot_results(X_scaled, processed_df)
        print("Plotting complete")
    except Exception as e:
        print("Error during plotting:", e)
        traceback.print_exc()
        sys.exit(1)

# Save Model and Preprocessing objects for new data
    try:
        joblib.dump(best_model, 'best_iso_forest_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(le_encoders, 'label_encoders.pkl')
        print("Model and preprocessing tools saved successfully!")
    except Exception as e:
        print("Error saving model or preprocessors:", e)
        traceback.print_exc()
        sys.exit(1)