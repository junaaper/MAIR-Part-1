# Evaluate the trained classifiers on one or more .dat files
# Usage: python evaluate.py --data /path/to/files.dat [--model all]
import os
import sys
import argparse
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

from dialog_acts import DATA_FILE, load_data, original_split, clean_utterance
from classifiers import BOW_MODELS, EMBEDDING_MODELS, load_model, predict

# A small test file doesn't have to contain every class, this is expected so no need to warn about it
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def evaluate_file(path, model_names, y_train):
    utterances, keywords = load_data(path)
    if len(utterances) == 0:
        print(f"Error: no utterances found in '{path}'")
        sys.exit(1)
    utterances = [clean_utterance(u) for u in utterances]
    print(f"\n##### {os.path.basename(path)}: {len(utterances)} utterances #####")

    # Labels the models have never seen can't be predicted, so warn about them
    unknown_labels = set(keywords) - set(y_train)
    if unknown_labels:
        print("Warning: labels not seen during training:", ", ".join(sorted(unknown_labels)))

    results = []
    for name in model_names:
        model = load_model(name)
        if model is None:
            if name in EMBEDDING_MODELS:
                print(f"\nSkipping {name}: not trained (run python train.py without --skip-embeddings)")
                continue
            print(f"Error: model '{name}' not found, run python train.py first")
            sys.exit(1)

        y_pred = predict(name, model, utterances)

        print(f"\n=== {name} ===")
        print(classification_report(keywords, y_pred, zero_division=0))

        results.append({
            "model": name,
            "accuracy": accuracy_score(keywords, y_pred),
            "balanced_accuracy": balanced_accuracy_score(keywords, y_pred),
            "macro_f1": f1_score(keywords, y_pred, average="macro", zero_division=0),
        })

    print(f"\n=== Summary for {os.path.basename(path)} ===")
    print(pd.DataFrame(results).to_string(index=False, float_format="%.2f"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the dialog act classifiers on .dat files.")
    parser.add_argument("--data", nargs="+", required=True, help="one or more .dat files to evaluate")
    parser.add_argument("--model", default="all", choices=["all"] + BOW_MODELS + EMBEDDING_MODELS,
                        help="which classifier to evaluate (default: all trained models)")
    args = parser.parse_args()

    _, _, y_train, _ = original_split(*load_data(DATA_FILE))
    model_names = BOW_MODELS + EMBEDDING_MODELS if args.model == "all" else [args.model]
    for path in args.data:
        evaluate_file(path, model_names, y_train)


if __name__ == "__main__":
    main()
