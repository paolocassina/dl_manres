import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
import os
from datetime import datetime



def load_data(embedding_file, label_file):
    """Loads verb embeddings and their Manner/Result labels."""
    # Load embeddings
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)

    # Load labels
    labels_df = pd.read_csv(label_file, index_col=0)  # Verb lemma as index
    labels = labels_df["class"].map({"manner": 0, "result": 1})  # Convert labels to binary

    X, y = [], []
    # Ensure embeddings match labels
    for verb, embedding in embeddings.items():
        if verb in list(labels.index):
            X.append(embedding)
            print(labels[verb])
            y.append(labels[verb])

        else:
            print(f"{verb} not included")
    return np.array(X), np.array(y)


def apply_pca(X, n_components=53):
    """Reduces embeddings to n_components using PCA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_components = min(X.shape[0], X.shape[1])
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_.sum()  # Return variance explained

def evaluate_model(model, X, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)

    metrics = {"accuracy": [], "f1": [], "auc_roc":[]}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # compute metrics
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["auc_roc"].append(roc_auc_score(y_test, y_prob))

    return {metric:np.mean(values) for metric, values in metrics.items()}

def random_baseline(y):
    np.random.seed(42)
    y_random = np.random.permutation(y)

    return y_random


def save_results(embs_model, embs_layer, model_type, results, pca_info=None, random_baseline_results=None, output_dir="../results/probe"):
    """Saves evaluation metrics to a timestamped text file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"probe_results_{model_type}_{timestamp}.txt")

    with open(filename, "w") as f:
        f.write(f"Model:{(str.upper(embs_model))}, Layer: {embs_layer}")
        f.write(f"Probing Analysis Results ({model_type.upper()})\n")
        f.write("="*40 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("Model: Logistic Regression\n")
        f.write("Cross-validation: 5-fold stratified\n\n")

        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1-score: {results['f1']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n\n")

        if pca_info:
            f.write(f"PCA Applied\n")
            f.write(f"PCA Info: {pca_info:.4f}\n")
            # f.write(f"F1-score: {pca_info['f1']:.4f}\n")
            # f.write(f"AUC-ROC: {pca_info['auc_roc']:.4f}\n")


        if random_baseline_results:
            f.write("Random Baseline Results (Shuffled Labels)\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy: {random_baseline_results['accuracy']:.4f}\n")
            f.write(f"F1-score: {random_baseline_results['f1']:.4f}\n")
            f.write(f"AUC-ROC: {random_baseline_results['auc_roc']:.4f}\n")

    print(f"Results saved to {filename}")


def main(embedding_file, label_file, pca=False):
    """Runs probing analysis with optional PCA reduction."""
    print(f"Loading data from {embedding_file} and {label_file}...")
    X, y = load_data(embedding_file, label_file)

    embs_model = embedding_file.split("_")[0]
    embs_layer = embedding_file.split("_")[1]

    # Run Standard Logistic Regression
    print("Running Standard Logistic Regression probing with 5-fold CV...")
    model = LogisticRegression(solver="liblinear", random_state=42)
    results = evaluate_model(model, X, y)

    print("\nStandard Logistic Regression Results:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Run dummy classification (shuffled labels) on Full-Dimensional Data
    print("\nRunning Random Baseline (Shuffled Labels) on Full-Dimensional Data...")
    y_random = random_baseline(y)
    random_baseline_results = evaluate_model(model, X, y_random)

    print("\nRandom Baseline Results:")
    for metric, value in random_baseline_results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Save Standard results
    save_results(embs_model, embs_layer,"logistic", results, None, random_baseline_results)

    # If PCA is enabled, apply it and run logistic regression again
    if pca:
        print(f"\nApplying PCA reduction to {min(X.shape[0], X.shape[1])} dimensions...")
        X_pca, variance_explained = apply_pca(X)
        print("Running Logistic Regression on PCA-reduced data...")
        # print(f"Variance explained: {variance_explained:.4f}")
        pca_results = evaluate_model(model, X_pca, y)

        print("\nLogistic Regression on PCA-Reduced Data Results:")
        for metric, value in pca_results.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        # Save PCA results separately
        save_results("logistic_pca", pca_results, variance_explained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_file", type=str, default="../data/embs/bert-base-uncased_6_verb_embeddings.pkl",
                        help="Path to the embeddings file")
    parser.add_argument("--label_file", type=str, default="../data/verbs/manres_lemma.csv", help="Path to the labels file")

    args = parser.parse_args()
    main(args.embedding_file, args.label_file, args.pca)