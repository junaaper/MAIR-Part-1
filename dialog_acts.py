# Loading the dialog act data and making the original and grouped splits
import os
import re
import sys
import random
from collections import Counter

from sklearn.model_selection import train_test_split


SEED = 12345 #global
TEST_FRAC = 0.15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "dailog_acts.dat") # The file is wrongly spelled in the assignment download
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_data(data_file):
    """Read a .dat file (dialog_act [space] utterance) and return the utterances and keywords."""
    if not os.path.isfile(data_file):
        print(f"Error: could not find data file '{data_file}'")
        sys.exit(1)

    utterances = []
    keywords = []
    skipped = 0

    # Splitting the keywords from the sentences
    with open(data_file, "r", encoding="utf-8", errors="replace") as file:
        for utterance in file:
            u = utterance.lower().strip() # Convert the data to lower case for training and testing

            # Skip empty lines and lines with only a label, so an external file can't crash the split
            if len(u.split(maxsplit=1)) < 2:
                if u:
                    skipped += 1
                continue

            keyword_u, sentence_u = u.split(maxsplit=1)

            utterances.append(sentence_u)
            keywords.append(keyword_u)

    if skipped > 0:
        print(f"Warning: skipped {skipped} line(s) without an utterance in {data_file}")

    return utterances, keywords


def original_split(utterances, keywords):
    """Stratified random 85/15 split (Original split)."""
    # Don't split again, we need the same split for each classifier
    X_train, X_test, y_train, y_test = train_test_split(utterances, keywords, test_size=TEST_FRAC, stratify=keywords, random_state=SEED)
    return X_train, X_test, y_train, y_test


def grouped_split(utterances, keywords):
    """85/15 split where all duplicates of an utterance end up in the same split (Grouped split)."""
    rng = random.Random(SEED)

    # Group rows by unique utterance, tag with majority class + size
    utterance_groups = {}
    for s, k in zip(utterances, keywords):
        utterance_groups.setdefault(s, []).append(k)

    groups_by_class = {}
    for s, klist in utterance_groups.items():
        majority = Counter(klist).most_common(1)[0][0]
        groups_by_class.setdefault(majority, []).append((s, len(klist)))

    test_utterances = set()

    for cls, groups in groups_by_class.items():
        total_rows = sum(size for _, size in groups)
        target = TEST_FRAC * total_rows
        rng.shuffle(groups)
        running_test = 0
        groups = sorted(groups, key=lambda g: g[1])   # Smallest group first
        for i, (s, size) in enumerate(groups):
            take = (i == 0) or abs(running_test + size - target) <= abs(running_test - target)
            if take:
                test_utterances.add(s)
                running_test += size

    X_train_gs, y_train_gs, X_test_gs, y_test_gs = [], [], [], []
    for s, k in zip(utterances, keywords):
        if s in test_utterances:
            X_test_gs.append(s); y_test_gs.append(k)
        else:
            X_train_gs.append(s); y_train_gs.append(k)

    return X_train_gs, X_test_gs, y_train_gs, y_test_gs


def clean_utterance(text):
    """Lower case user input and keep only word tokens (same token pattern as the vectorizers)."""
    return " ".join(re.findall(r"\b\w+\b", text.lower()))
