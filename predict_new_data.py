import os
import pandas as pd
import joblib


def predict_anomalies(new_data_path, model_dir=None, save_path=None):
    """Load saved model/scalers/encoders, predict anomalies for `new_data_path`.

    Args:
        new_data_path (str): path to the new CSV file to predict on.
        model_dir (str, optional): directory where saved pickles live. Defaults to the
            script directory.
        save_path (str, optional): if provided, save resulting DataFrame to this CSV.

    Returns:
        pd.DataFrame: the input DataFrame with `is_anomaly` and `anomaly_label` columns.
    """
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load saved components
    try:
        model = joblib.load(os.path.join(model_dir, 'best_iso_forest_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        le_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required model file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading model components: {e}")

    # 2. Load and clean new data
    df = pd.read_csv(new_data_path)

    # Ensure duration is numeric (remove commas if present)
    if 'duration' in df.columns:
        df['duration'] = df['duration'].astype(str).str.replace(',', '', regex=False)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # 3. Apply stored encoding
    features = ['duration', 'charge', 'city', 'destination_type', 'call_direction']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in new data: {missing}")

    X_new = df[features].copy()

    for col, le in le_encoders.items():
        # Fill NaNs with a known class and map unseen values to a default class
        X_new[col] = X_new[col].fillna(le.classes_[0])
        X_new[col] = X_new[col].where(X_new[col].isin(le.classes_), le.classes_[0])
        try:
            X_new[col] = le.transform(X_new[col])
        except Exception as e:
            raise RuntimeError(f"Error transforming column '{col}': {e}")

    # 4. Scale and Predict
    try:
        X_scaled = scaler.transform(X_new)
        preds = model.predict(X_scaled)
    except Exception as e:
        raise RuntimeError(f"Error during scaling/prediction: {e}")

    df['is_anomaly'] = preds
    df['anomaly_label'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    # Save results if requested
    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved predictions to: {save_path}")

    return df


# Example Usage (uncomment and modify paths to run):
# results = predict_anomalies('new_df.csv', model_dir='.', save_path='predictions.csv')
# print(results[results['anomaly_label'] == 'Anomaly'].head())