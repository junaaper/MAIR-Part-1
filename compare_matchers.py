# Compare the two slot extraction fallbacks (Levenshtein vs semantic similarity) on data/matcher_tests.txt
# Usage: python compare_matchers.py [--sweep]
import os
import time
import argparse

import numpy as np
import pandas as pd

import slot_extraction
from dialog_acts import BASE_DIR, RESULTS_DIR

TEST_FILE = os.path.join(BASE_DIR, "data", "matcher_tests.txt")


def load_tests(path=TEST_FILE):
    """Read the tab separated test file, returns a list of dicts."""
    tests = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            category, asked, expected, utterance = line.rstrip("\n").split("\t")
            slot, value = expected.split("=") if expected != "none" else (None, None)
            tests.append({"category": category, "asked": None if asked == "-" else asked,
                          "slot": slot, "value": value, "expected": expected, "utterance": utterance})
    return tests


def predicted_values(extraction):
    """All slot values found (certain and to be confirmed), dontcare not included."""
    values = {slot: value for slot, value in extraction.slots.items() if value != "dontcare"}
    values.update({slot: value for slot, (_, value) in extraction.uncertain.items()})
    return values


def run(tests, matcher):
    """Run extraction on every test, returns one row per test."""
    rows = []
    for t in tests:
        start = time.perf_counter()
        extraction = slot_extraction.extract(t["utterance"], t["asked"], matcher)
        ms = (time.perf_counter() - start) * 1000
        found = predicted_values(extraction)

        if t["slot"] is None:
            outcome = "correct" if not found else "false match"
            prediction = ", ".join(f"{s}={v}" for s, v in found.items())
        elif found.get(t["slot"]) == t["value"]:
            outcome, prediction = "correct", t["value"]
        elif t["slot"] in found:
            outcome, prediction = "false match", found[t["slot"]]
        else:
            outcome, prediction = "no match", ""
        rows.append({"matcher": matcher, **t, "prediction": prediction, "outcome": outcome, "ms": ms})
    return rows


def summarize(results):
    """Accuracy, false matches, no matches and speed per matcher and category."""
    summary = []
    for (matcher, category), group in results.groupby(["matcher", "category"], sort=False):
        summary.append({"matcher": matcher, "category": category, "n": len(group),
                        "accuracy": (group["outcome"] == "correct").mean(),
                        "false_matches": int((group["outcome"] == "false match").sum()),
                        "no_matches": int((group["outcome"] == "no match").sum())})
    for matcher, group in results.groupby("matcher", sort=False):
        summary.append({"matcher": matcher, "category": "ALL", "n": len(group),
                        "accuracy": (group["outcome"] == "correct").mean(),
                        "false_matches": int((group["outcome"] == "false match").sum()),
                        "no_matches": int((group["outcome"] == "no match").sum()),
                        "ms_per_utterance": group["ms"].median()})
    return pd.DataFrame(summary)


def sweep(tests):
    """Accuracy of the semantic matcher for different thresholds, used to choose SEMANTIC_THRESHOLD."""
    original = slot_extraction.SEMANTIC_THRESHOLD
    rows = []
    for threshold in np.arange(0.20, 0.80, 0.05):
        slot_extraction.SEMANTIC_THRESHOLD = threshold
        results = pd.DataFrame(run(tests, "semantic"))
        rows.append({"threshold": round(threshold, 2), "accuracy": (results["outcome"] == "correct").mean(),
                     "false_matches": int((results["outcome"] == "false match").sum()),
                     "no_matches": int((results["outcome"] == "no match").sum())})
    slot_extraction.SEMANTIC_THRESHOLD = original
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare the Levenshtein and semantic slot matchers.")
    parser.add_argument("--sweep", action="store_true", help="also try different semantic thresholds")
    args = parser.parse_args()

    tests = load_tests()
    print(f"{len(tests)} test utterances, loading DistilBERT...")
    slot_extraction.warm_up()

    results = pd.DataFrame(run(tests, "levenshtein") + run(tests, "semantic"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "matcher_predictions.csv"), index=False)

    print("\n=== Predictions ===")
    table = results.pivot_table(index=["category", "utterance", "expected"], columns="matcher",
                                values="prediction", aggfunc="first", sort=False).fillna("")
    print(table.to_string())

    summary = summarize(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "matcher_comparison.csv"), index=False)
    print("\n=== Summary (semantic threshold", slot_extraction.SEMANTIC_THRESHOLD, ") ===")
    print(summary.to_string(index=False, float_format="%.2f", na_rep=""))

    if args.sweep:
        print("\n=== Semantic threshold sweep ===")
        print(sweep(tests).to_string(index=False, float_format="%.2f"))

    print("\nSaved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
