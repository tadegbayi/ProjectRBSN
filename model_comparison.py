import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def load_sample(path='January_masked_sample.csv', n_samples=20000, random_state=42):
    df = pd.read_csv(path)
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=random_state).copy()
    # clean numeric columns
    if 'duration' in df.columns:
        df['duration'] = df['duration'].astype(str).str.replace(',', '', regex=False)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    if 'charge' in df.columns:
        df['charge'] = pd.to_numeric(df['charge'], errors='coerce')
    return df


def prepare_features(df, features=None):
    if features is None:
        features = ['duration', 'charge', 'city', 'destination_type', 'call_direction']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[features].copy()
    # encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler, X


def run_models(Xs, contamination):
    results = {}
    # IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso.fit_predict(Xs)
    results['IsolationForest'] = iso_labels

    # LOF (use fit_predict)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_labels = lof.fit_predict(Xs)
    results['LOF'] = lof_labels

    # OneClassSVM
    oc = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    oc_labels = oc.fit_predict(Xs)
    results['OneClassSVM'] = oc_labels

    return results


def jaccard(a, b):
    set_a = set(np.where(a == -1)[0])
    set_b = set(np.where(b == -1)[0])
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union) if union else 0.0


def main():
    out = 'model_comparison'
    os.makedirs(out, exist_ok=True)
    df = load_sample()
    Xs, scaler, Xraw = prepare_features(df)

    contaminations = [0.001, 0.005, 0.01, 0.02, 0.05]
    records = []
    jac_rows = []

    for c in contaminations:
        res = run_models(Xs, contamination=c)
        counts = {m: int((labels == -1).sum()) for m, labels in res.items()}
        for m, cnt in counts.items():
            records.append({'model': m, 'contamination': c, 'n_anomalies': cnt})

        # jaccard pairwise
        jac_if_lof = jaccard(res['IsolationForest'], res['LOF'])
        jac_if_oc = jaccard(res['IsolationForest'], res['OneClassSVM'])
        jac_lof_oc = jaccard(res['LOF'], res['OneClassSVM'])
        jac_rows.append({'contamination': c, 'IF_LOF': jac_if_lof, 'IF_OC': jac_if_oc, 'LOF_OC': jac_lof_oc})

        # save PCA plots for c=0.01
        if abs(c - 0.01) < 1e-12:
            pca = PCA(n_components=2, random_state=42)
            Xp = pca.fit_transform(Xs)
            # downsample for plotting to keep rendering fast
            n_plot = min(5000, Xp.shape[0])
            idx = np.random.RandomState(42).choice(Xp.shape[0], size=n_plot, replace=False)
            methods = ['IsolationForest', 'LOF', 'OneClassSVM']
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for i, m in enumerate(methods):
                axes[i].scatter(Xp[idx, 0], Xp[idx, 1], c=(res[m][idx] == -1), cmap='coolwarm', s=8, alpha=0.7)
                axes[i].set_title(f'{m} (cont={c})')
            plt.savefig(os.path.join(out, 'pca_models_cont_0.01.png'), bbox_inches='tight')
            plt.close()

    df_counts = pd.DataFrame(records)
    df_jac = pd.DataFrame(jac_rows)
    df_counts.to_csv(os.path.join(out, 'model_counts.csv'), index=False)
    df_jac.to_csv(os.path.join(out, 'jaccard_similarity.csv'), index=False)

    # Plot anomaly counts per model
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_counts, x='contamination', y='n_anomalies', hue='model', marker='o')
    plt.xscale('log')
    plt.xlabel('contamination')
    plt.ylabel('n_anomalies')
    plt.title('Anomalies detected vs contamination (log scale)')
    plt.savefig(os.path.join(out, 'anomaly_counts_vs_contamination.png'), bbox_inches='tight')
    plt.close()

    # Plot Jaccard similarities
    df_jac_melt = df_jac.melt(id_vars='contamination', var_name='pair', value_name='jaccard')
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_jac_melt, x='contamination', y='jaccard', hue='pair', marker='o')
    plt.xscale('log')
    plt.xlabel('contamination')
    plt.ylabel('Jaccard similarity')
    plt.title('Pairwise Jaccard similarity of anomaly sets')
    plt.savefig(os.path.join(out, 'jaccard_vs_contamination.png'), bbox_inches='tight')
    plt.close()

    print('Model comparison artifacts saved to', out)


if __name__ == '__main__':
    main()
