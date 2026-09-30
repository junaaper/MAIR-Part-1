# Evaluation for the report: metrics, error analysis, difficult cases and the influence of data leakage
# Usage: python analysis.py [--skip-embeddings]
import io
import os
import argparse
import warnings
from collections import Counter
from contextlib import redirect_stdout

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

from dialog_acts import DATA_FILE, MODELS_DIR, RESULTS_DIR, BASE_DIR, load_data, original_split, grouped_split
from classifiers import classify, train_and_eval_svm, train_and_eval_mlp, train_and_eval_dt
from classifiers import train_and_eval_mlp_embeddings, train_and_eval_svm_emb

# The grouped test set contains classes that are (almost) missing from its train set, that is part of the analysis
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

EMBEDDING_CACHE = os.path.join(MODELS_DIR, "embedding_cache.joblib")
DIFFICULT_SETS = {
    "negation": os.path.join(BASE_DIR, "data", "difficult_negation.dat"),
    "disfluent": os.path.join(BASE_DIR, "data", "difficult_disfluent.dat"),
}


class Embedder:
    """DistilBERT embeddings per unique utterance, cached on disk (only ~4800 of the 24000 utterances are unique)."""

    def __init__(self):
        self.cache = joblib.load(EMBEDDING_CACHE) if os.path.isfile(EMBEDDING_CACHE) else {}

    def __call__(self, utterances):
        missing = sorted(set(utterances) - set(self.cache))
        if missing:
            from embeddings import get_embeddings
            for u, vector in zip(missing, get_embeddings(missing)):
                self.cache[u] = vector
            joblib.dump(self.cache, EMBEDDING_CACHE)
        return np.array([self.cache[u] for u in utterances])


def train_models(X_train, y_train, X_test, y_test, embed):
    """Train every classifier on one split, return a predict function and the predictions on its test set."""
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    models = {}
    predictions = {"rule": [classify(u) for u in X_test]}
    with redirect_stdout(io.StringIO()):
        models["svm"], predictions["svm"] = train_and_eval_svm(X_train_bow, y_train, X_test_bow, y_test, "")
        models["mlp"], predictions["mlp"] = train_and_eval_mlp(X_train_bow, y_train, X_test_bow, y_test, "")
        models["dt"], predictions["dt"] = train_and_eval_dt(X_train_bow, y_train, X_test_bow, y_test, "")
        if embed is not None:
            X_train_emb, X_test_emb = embed(X_train), embed(X_test)
            models["svm_emb"], predictions["svm_emb"] = train_and_eval_svm_emb(X_train_emb, y_train, X_test_emb, y_test, "")
            models["mlp_emb"], predictions["mlp_emb"] = train_and_eval_mlp_embeddings(X_train_emb, y_train, X_test_emb, y_test, "")
            models["dt_emb"], predictions["dt_emb"] = train_and_eval_dt(X_train_emb, y_train, X_test_emb, y_test, "")

    def predict_all(utterances):
        result = {"rule": [classify(u) for u in utterances]}
        bow = vectorizer.transform(utterances)
        emb = embed(utterances) if embed is not None else None
        for name, model in models.items():
            result[name] = list(model.predict(emb if name.endswith("_emb") else bow))
        return result

    predictions = {name: list(p) for name, p in predictions.items()}
    return predict_all, predictions, vectorizer


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def show(title, table):
    print(f"\n=== {title} ===")
    print(table.to_string(index=False, float_format="%.3f"))


def main():
    parser = argparse.ArgumentParser(description="Evaluation for the report (metrics, errors, difficult cases, leakage).")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="only use the rule based and bag-of-words models")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    embed = None if args.skip_embeddings else Embedder()

    utterances, keywords = load_data(DATA_FILE)
    splits = {
        "original": original_split(utterances, keywords),
        "grouped": grouped_split(utterances, keywords),
    }

    trained = {}
    for split, (X_train, X_test, y_train, y_test) in splits.items():
        print(f"Training all models on the {split} split...")
        trained[split] = train_models(X_train, y_train, X_test, y_test, embed)

    # 1. Quantitative evaluation
    rows = []
    for split, (X_train, X_test, y_train, y_test) in splits.items():
        majority = Counter(y_train).most_common(1)[0][0]
        rows.append({"model": "majority", "split": split, **metrics(y_test, [majority] * len(y_test))})
        for name, y_pred in trained[split][1].items():
            rows.append({"model": name, "split": split, **metrics(y_test, y_pred)})
    metrics_table = pd.DataFrame(rows)
    show("Metrics per model and split", metrics_table)
    metrics_table.to_csv(os.path.join(RESULTS_DIR, "eval_metrics.csv"), index=False)

    # 2. Error analysis on the grouped split
    X_train, X_test, y_train, y_test = splits["grouped"]
    _, predictions, vectorizer = trained["grouped"]
    classes = sorted(set(keywords))

    train_counts, test_counts = Counter(y_train), Counter(y_test)
    per_class = {"dialog_act": classes,
                 "train_count": [train_counts[c] for c in classes],
                 "test_count": [test_counts[c] for c in classes]}
    for name, y_pred in predictions.items():
        per_class[name] = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)[2]
    per_class_table = pd.DataFrame(per_class).sort_values("train_count", ascending=False)
    show("F1 per dialog act (grouped split)", per_class_table)
    per_class_table.to_csv(os.path.join(RESULTS_DIR, "eval_per_class_f1.csv"), index=False)

    confusions = Counter()
    for name, y_pred in predictions.items():
        if name != "rule":
            confusions.update((t, p) for t, p in zip(y_test, y_pred) if t != p)
    confusion_table = pd.DataFrame([{"true": t, "predicted": p, "errors_summed_over_ml_models": n}
                                    for (t, p), n in confusions.most_common(15)])
    show("Most frequent confusions (grouped split, all ML models)", confusion_table)
    confusion_table.to_csv(os.path.join(RESULTS_DIR, "eval_confusions.csv"), index=False)

    # Utterances every system gets wrong, counted once per unique utterance
    hard = {}
    for i, (u, true) in enumerate(zip(X_test, y_test)):
        wrong = {name: y_pred[i] for name, y_pred in predictions.items() if y_pred[i] != true}
        if len(wrong) == len(predictions):
            entry = hard.setdefault(u, {"utterance": u, "true": true, "count": 0,
                                        "known_words": vectorizer.transform([u]).sum(), **wrong})
            entry["count"] += 1
    hard_table = pd.DataFrame(list(hard.values())).sort_values("count", ascending=False)
    print(f"\n{len(hard_table)} unique test utterances ({hard_table['count'].sum()} rows) are misclassified by every system")
    print("Per true dialog act:", dict(Counter(hard_table["true"])))
    show("Most frequent of these", hard_table.head(20))
    hard_table.to_csv(os.path.join(RESULTS_DIR, "eval_hard_utterances.csv"), index=False)

    # 3. Difficult cases
    predict_all = trained["original"][0]
    X_train_original = set(splits["original"][0])
    difficult_rows = []
    for set_name, path in DIFFICULT_SETS.items():
        set_utterances, set_keywords = load_data(path)
        for u in set_utterances:
            if u in X_train_original:
                print(f"Warning: '{u}' from {set_name} appears in the training data")
        set_predictions = predict_all(set_utterances)
        for name, y_pred in set_predictions.items():
            difficult_rows.append({"set": set_name, "model": name, "accuracy": accuracy_score(set_keywords, y_pred)})
        pd.DataFrame({"utterance": set_utterances, "true": set_keywords, **set_predictions}).to_csv(
            os.path.join(RESULTS_DIR, f"eval_difficult_{set_name}.csv"), index=False)
    difficult_table = pd.DataFrame(difficult_rows).pivot(index="model", columns="set", values="accuracy").reset_index()
    show("Accuracy on the difficult cases (original split models)", difficult_table)
    difficult_table.to_csv(os.path.join(RESULTS_DIR, "eval_difficult.csv"), index=False)

    # 4. Data leakage: how many test sentences were already in the train set, and how the models do on each part
    leakage_rows = []
    for split, (X_train, X_test, y_train, y_test) in splits.items():
        train_set = set(X_train)
        seen_in_train = np.array([u in train_set for u in X_test])
        for name, y_pred in trained[split][1].items():
            y_true, y_pred = np.array(y_test), np.array(y_pred)
            row = {"split": split, "model": name, "test_seen_in_train": seen_in_train.mean()}
            for part, mask in [("seen", seen_in_train), ("unseen", ~seen_in_train)]:
                row[f"accuracy_{part}"] = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else np.nan
            leakage_rows.append(row)
    leakage_table = pd.DataFrame(leakage_rows)
    show("Accuracy on test sentences seen / unseen during training", leakage_table)
    leakage_table.to_csv(os.path.join(RESULTS_DIR, "eval_leakage.csv"), index=False)

    print("\nSaved eval_*.csv to", RESULTS_DIR)


if __name__ == "__main__":
    main()
