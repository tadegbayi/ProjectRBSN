# ProjectRBSN — Anomaly Detection for CDRs

This repository contains scripts to train/run anomaly detection on call detail records (CDRs) and to run predictions on new data.

**Files**
- `anomaly_detection.py`: training and visualization script that prepares data, runs IsolationForest, LOF and OneClassSVM, and saves model/preprocessing artifacts.
- `predict_new_data.py`: utility to load saved model/scaler/label-encoders and predict anomalies on new CSV files (adds `is_anomaly` and `anomaly_label`).
- `January_masked_sample.csv`: example dataset used for training.
- `new_df.csv`: example new data used for prediction.
- `best_iso_forest_model.pkl`, `scaler.pkl`, `label_encoders.pkl`: saved artifacts produced by `anomaly_detection.py`.
- `predictions.csv`: example output produced by `predict_new_data.predict_anomalies()`.

**Requirements**
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Install dependencies:
```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**Usage — training / exploring**
1. Edit `anomaly_detection.py` paths if needed (default path uses `C:/Users/HP/Downloads/ProjectRBSN/January_masked_sample.csv`).
2. Run training + visualization:
```bash
python anomaly_detection.py
```
Outputs:
- `anomaly_comparison_local.png` — PCA scatterplots of model labels.
- `best_iso_forest_model.pkl`, `scaler.pkl`, `label_encoders.pkl` — artifacts saved for later predictions.

**Usage — predict on new data**
Use the helper in `predict_new_data.py` to run predictions and optionally save results to CSV.

From Python code:
```python
from predict_new_data import predict_anomalies
results = predict_anomalies('new_df.csv', model_dir='.', save_path='predictions.csv')
print(results[results['anomaly_label'] == 'Anomaly'])
```

Or from the shell:
```bash
python -c "from predict_new_data import predict_anomalies; predict_anomalies('new_df.csv', save_path='predictions.csv')"
```

**What the output contains**
- `is_anomaly`: model output (1 for inlier / -1 for outlier).
- `anomaly_label`: human-readable label (`Normal` / `Anomaly`).

**Notes & recommendations**
- File paths: use forward slashes (`/`) or raw strings on Windows to avoid escape issues.
- Encoding: `predict_new_data` maps unseen categorical values to the first class learned by the `LabelEncoder`. Consider updating this behavior if you prefer a different fallback (e.g., `Unknown` or retraining encoders).
- Missing columns in new data will raise an error — ensure `duration`, `charge`, `city`, `destination_type`, `call_direction` exist.
- If you re-run training with a different preprocessing pipeline, remember to update the saved artifacts and `model_dir` used by `predict_new_data`.

**Troubleshooting**
- Unicode/escape errors when specifying Windows paths: use `C:/path/to/file.csv` or prefix with `r"C:\path\to\file.csv"`.
- `FileNotFoundError` when loading model artifacts: ensure the `.pkl` files are present in `model_dir` or pass the correct `model_dir` to `predict_anomalies()`.
- If LOF warns about duplicate values, increase `n_neighbors` in `anomaly_detection.py`.

**Next steps / suggestions**
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add unit tests or small example notebooks demonstrating end-to-end prediction.
- Decide on a policy for unseen categorical labels (e.g., map to `Unknown`, extend encoders, or use `OneHotEncoder` with `handle_unknown='ignore'`).

---
Generated README for sharing with teammates. Edit as needed.
