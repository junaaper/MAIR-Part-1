# Train and evaluate all dialog act classifiers on the original and grouped split
import os
import argparse

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Only save the plot, don't open a window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from dialog_acts import DATA_FILE, MODELS_DIR, RESULTS_DIR, load_data, original_split, grouped_split
from classifiers import RULES, classify, train_and_eval_svm, train_and_eval_mlp, train_and_eval_dt
from classifiers import train_and_eval_mlp_embeddings, train_and_eval_svm_emb


def plot_grouped_bars(ax, models, data_dict, title, ylabel):
    """
    Renders a grouped bar chart on a Matplotlib Axes object.

    ax: Matplotlib Axes object to draw on.
    models: List of category labels for the x-axis.
    data_dict: Dictionary mapping legend labels to data arrays.
    """
    x_indexes = np.arange(len(models))
    num_groups = len(data_dict)

    # Calculate bar width to fit 0.8 unit space
    bar_width = 0.8 / num_groups

    # Calculate the starting offset so the cluster of bars is centered on the tick
    offset = (num_groups - 1) / 2 * bar_width
    colors = ['#0072B2', '#56B4E9', '#D55E00', '#E69F00']

    for i, (label, values) in enumerate(data_dict.items()):
        # Shift each successive bar to the right
        pos = x_indexes - offset + (i * bar_width)
        ax.bar(pos, values, width=bar_width, label=label, color=colors[i % len(colors)])

    # Formatting
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1) # Standardize Y-axis from 0 to 1.1 to leave room for the legend
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def add_result(all_results, model, features, split, y_true, y_pred):
    """Store accuracy and balanced accuracy of one model/split in the overall results list."""
    all_results.append({
        "model": model,
        "features": features,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    })


def main():
    parser = argparse.ArgumentParser(description="Train all dialog act classifiers.")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="only train the rule based and bag-of-words models (no torch/DistilBERT needed)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    # Convert the data for training and testing
    utterances, keywords = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = original_split(utterances, keywords)
    X_train_gs, X_test_gs, y_train_gs, y_test_gs = grouped_split(utterances, keywords)

    print(len(X_train), "original train size")
    print(len(X_test), "original test size")
    print(len(X_train_gs), "grouped train size")
    print(len(X_test_gs), "grouped test size")

    # Rule based
    results = []
    for u in utterances:
        prediction = classify(u)
        results.append(prediction)

    classses = [k for k, _, _ in RULES]
    print("results on entire dataset:")
    print(classification_report(keywords, results, labels=classses, zero_division=0))

    results_test = []
    for u_test in X_test:
        prediction = classify(u_test)
        results_test.append(prediction)

    print("results on normal split test set:")
    print(classification_report(y_test, results_test, labels=classses, zero_division=0))

    results_test_gs = []
    for u_test_gs in X_test_gs:
        prediction = classify(u_test_gs)
        results_test_gs.append(prediction)

    print("results on grouped split test set:")
    print(classification_report(y_test_gs, results_test_gs, labels=classses, zero_division=0))

    add_result(all_results, "rule", "keywords", "original", y_test, results_test)
    add_result(all_results, "rule", "keywords", "grouped", y_test_gs, results_test_gs)

    # Bag-of-words for Original and Grouped split

    # Original split
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_train_bow = vectorizer.fit_transform(X_train)   # Learn vocabulary from TRAIN only
    X_test_bow = vectorizer.transform(X_test)         # Unknown (OOV) words are dropped

    # Grouped split
    vectorizer_gs = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_train_gs_bow = vectorizer_gs.fit_transform(X_train_gs)
    X_test_gs_bow = vectorizer_gs.transform(X_test_gs)

    print("Vocabulary size (original):", len(vectorizer.vocabulary_))
    print("Vocabulary size (grouped):", len(vectorizer_gs.vocabulary_))

    # SVM
    # Train and test on both splits
    svm, y_pred_svm = train_and_eval_svm(
        X_train_bow, y_train, X_test_bow, y_test, "ORIGINAL split")

    svm_gs, y_pred_svm_gs = train_and_eval_svm(
        X_train_gs_bow, y_train_gs, X_test_gs_bow, y_test_gs, "GROUPED split")

    # Save the results
    svm_results = pd.DataFrame([{
        "svm_original_accuracy": accuracy_score(y_test, y_pred_svm),
        "svm_original_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_svm),
        "svm_grouped_accuracy": accuracy_score(y_test_gs, y_pred_svm_gs),
        "svm_grouped_balanced_accuracy": balanced_accuracy_score(y_test_gs, y_pred_svm_gs),
    }])
    print(svm_results)
    svm_results.to_csv(os.path.join(RESULTS_DIR, "svm_results.csv"), index=False)

    add_result(all_results, "svm", "bow", "original", y_test, y_pred_svm)
    add_result(all_results, "svm", "bow", "grouped", y_test_gs, y_pred_svm_gs)

    # FeedForward network
    mlp, y_pred_mlp = train_and_eval_mlp(
        X_train_bow, y_train, X_test_bow, y_test, "MLP original split")

    mlp_gs, y_pred_mlp_gs = train_and_eval_mlp(
        X_train_gs_bow, y_train_gs, X_test_gs_bow, y_test_gs, "MLP grouped split")

    # Save MLP results
    results = {
        "mlp_original_accuracy": accuracy_score(y_test, y_pred_mlp),
        "mlp_original_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_mlp),
        "mlp_grouped_accuracy": accuracy_score(y_test_gs, y_pred_mlp_gs),
        "mlp_grouped_balanced_accuracy": balanced_accuracy_score(y_test_gs, y_pred_mlp_gs)
    }

    results_table = pd.DataFrame([results])
    print(results_table)

    results_table.to_csv(os.path.join(RESULTS_DIR, "mlp_results.csv"), index=False)

    add_result(all_results, "mlp", "bow", "original", y_test, y_pred_mlp)
    add_result(all_results, "mlp", "bow", "grouped", y_test_gs, y_pred_mlp_gs)

    # Decision Tree

    # Original split
    dt_classifier, y_pred_dt = train_and_eval_dt(
        X_train_bow,
        y_train,
        X_test_bow,
        y_test,
        "Decision Tree - Original BoW split")

    # Grouped split
    dt_classifier_gs, y_pred_dt_gs = train_and_eval_dt(
        X_train_gs_bow,
        y_train_gs,
        X_test_gs_bow,
        y_test_gs,
        "Decision Tree - Grouped BoW split")

    # Save Decision Tree results
    dt_results = {
        "dt_original_accuracy": accuracy_score(y_test, y_pred_dt),
        "dt_original_balanced_accuracy": balanced_accuracy_score(y_test, y_pred_dt),
        "dt_grouped_accuracy": accuracy_score(y_test_gs, y_pred_dt_gs),
        "dt_grouped_balanced_accuracy": balanced_accuracy_score(y_test_gs, y_pred_dt_gs)
    }

    dt_results_table = pd.DataFrame([dt_results])
    print(dt_results_table)
    dt_results_table.to_csv(os.path.join(RESULTS_DIR, "dt_results.csv"), index=False)

    add_result(all_results, "dt", "bow", "original", y_test, y_pred_dt)
    add_result(all_results, "dt", "bow", "grouped", y_test_gs, y_pred_dt_gs)

    # Save the original split BoW models, these are used by evaluate.py and chat.py
    joblib.dump({"vectorizer": vectorizer, "classifier": svm}, os.path.join(MODELS_DIR, "svm.joblib"))
    joblib.dump({"vectorizer": vectorizer, "classifier": mlp}, os.path.join(MODELS_DIR, "mlp.joblib"))
    joblib.dump({"vectorizer": vectorizer, "classifier": dt_classifier}, os.path.join(MODELS_DIR, "dt.joblib"))

    # DistilBERT embeddings
    if not args.skip_embeddings:
        from embeddings import get_embeddings

        X_train_emb    = get_embeddings(X_train)
        X_test_emb     = get_embeddings(X_test)

        X_train_gs_emb = get_embeddings(X_train_gs)
        X_test_gs_emb  = get_embeddings(X_test_gs)

        mlp_emb, y_pred_mlp_emb = train_and_eval_mlp_embeddings(
            X_train_emb, y_train, X_test_emb, y_test, "ORIGINAL split - DistilBERT + MLP")

        mlp_gs_emb, y_pred_mlp_gs_emb = train_and_eval_mlp_embeddings(
            X_train_gs_emb, y_train_gs, X_test_gs_emb, y_test_gs, "GROUPED split - DistilBERT + MLP")

        svm_emb, y_pred_svm_emb = train_and_eval_svm_emb(
            X_train_emb, y_train, X_test_emb, y_test, "ORIGINAL split")

        svm_gs_emb, y_pred_svm_gs_emb = train_and_eval_svm_emb(
            X_train_gs_emb, y_train_gs, X_test_gs_emb, y_test_gs, "GROUPED split")

        # Decision Tree DilstilBERT embeddings

        # Original split
        dt_classifier_emb, y_pred_dt_emb = train_and_eval_dt(
            X_train_emb,
            y_train,
            X_test_emb,
            y_test,
            "Decision Tree - Original DistilBERT split")

        # Grouped split
        dt_classifier_gs_emb, y_pred_dt_gs_emb = train_and_eval_dt(
            X_train_gs_emb,
            y_train_gs,
            X_test_gs_emb,
            y_test_gs,
            "Decision Tree - Grouped DistilBERT split")

        add_result(all_results, "svm", "embeddings", "original", y_test, y_pred_svm_emb)
        add_result(all_results, "svm", "embeddings", "grouped", y_test_gs, y_pred_svm_gs_emb)
        add_result(all_results, "mlp", "embeddings", "original", y_test, y_pred_mlp_emb)
        add_result(all_results, "mlp", "embeddings", "grouped", y_test_gs, y_pred_mlp_gs_emb)
        add_result(all_results, "dt", "embeddings", "original", y_test, y_pred_dt_emb)
        add_result(all_results, "dt", "embeddings", "grouped", y_test_gs, y_pred_dt_gs_emb)

        joblib.dump({"classifier": svm_emb}, os.path.join(MODELS_DIR, "svm_emb.joblib"))
        joblib.dump({"classifier": mlp_emb}, os.path.join(MODELS_DIR, "mlp_emb.joblib"))
        joblib.dump({"classifier": dt_classifier_emb}, os.path.join(MODELS_DIR, "dt_emb.joblib"))

    # Eval and comparison
    all_results_table = pd.DataFrame(all_results)
    print(all_results_table.to_string(index=False))
    all_results_table.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)

    # Plot all data
    models = ['SVM', 'MLP', 'DT']
    feature_types = [('bow', ''), ('embeddings', '-EMB')]
    if args.skip_embeddings:
        feature_types = [('bow', '')]

    # Build accuracy and balanced accuracy dicts, e.g. 'OG' -> [svm, mlp, dt] original split BoW
    acc_data = {}
    bal_acc_data = {}
    for split, split_name in [('original', 'OG'), ('grouped', 'GS')]:
        for features, suffix in feature_types:
            rows = all_results_table[(all_results_table["split"] == split) & (all_results_table["features"] == features)]
            rows = rows.set_index("model").loc[["svm", "mlp", "dt"]]
            acc_data[split_name + suffix] = rows["accuracy"].tolist()
            bal_acc_data[split_name + suffix] = rows["balanced_accuracy"].tolist()

    # Generate Subplots
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot Accuracy on the left (axes[0])
    plot_grouped_bars(axes[0], models, acc_data,
                      title='Model Accuracy by Split & Embeddings',
                      ylabel='Accuracy')

    # Plot Balanced Accuracy on the right (axes[1])
    plot_grouped_bars(axes[1], models, bal_acc_data,
                      title='Balanced Accuracy by Split & Embeddings',
                      ylabel='Balanced Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    print("Saved models to", MODELS_DIR, "and results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
