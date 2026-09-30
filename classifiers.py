# Rule based baseline and the SVM, MLP and Decision Tree classifiers
import os

import joblib
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from dialog_acts import SEED, MODELS_DIR


# Rule based

# Hand written rules based on looking in the data (exercise requirement), goal is a 0.8 accuracy
RULES = [
    ('ack', 7, ["okay", "well", "fine", "kay"]),
    ('affirm', 6, ['yes', "ye", 'right', 'correct', 'affirmative']),
    ('bye', 3, ['goodbye', 'cya', 'see you', 'bye']),#
    ('confirm', 15, ["is", "it", "does"]),
    ('deny', 10, ["wrong", "not"]),
    ('hello', 2, ["hi", "hello", "halo", ]),#
    ('inform', 14, ["cheap", "west", "east", "south", "north", "type", "restaurant", "moderate", "moderately", "priced", "town", "area", "italian", "indian", "chinese", "expensive", "food", "want", "center", "matter", "care"]), #
    ('negate', 8, ["no"]),
    ('null', 16, ["unintelligible"]),
    ('repeat', 5, ["back", "repeat", "again", ]), #
    ('reqalts', 11, ["about", "else"]),
    ('reqmore', 12, ["more"]),
    ('request', 13, ["phone", "address", "price", "post", "what"]),
    ('restart', 4, ["start", "over", "reset"]),
    ('thankyou', 1, ['thank you', 'thank', 'thanks', 'appreciate', 'cheers'])#
]


def classify(utterance):
    """Rule based classification: return the dialog act of the matched rule with the best priority."""
    found_keywords = [("null", 999)]

    u_words = utterance.split()

    for keywords, priority, rule in RULES:
        for keyword in rule:
            #multi-word keywords like "see you" are matched against the whole utterance
            if keyword in u_words or (" " in keyword and f" {keyword} " in f" {utterance} "):
                found_keywords.append((keywords, priority))
                break

    found_keywords.sort(key=lambda x: x[1])

    return found_keywords[0][0]


# SVM

# SVM function
def train_and_eval_svm(X_train, y_train, X_test, y_test, label):
    """Train a linear SVM on bag-of-words features and print its results on the test set."""
    svm = LinearSVC(max_iter=5000, random_state=SEED)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f"{label}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    return svm, y_pred


# FeedForward network

# Bag-of-words MLP, used for both the original and the grouped split
def train_and_eval_mlp(X_train_bow, y_train, X_test_bow, y_test, label):
    """Train an MLP on bag-of-words features and print its results on the test set."""
    n_oov_rows = int((X_test_bow.sum(axis=1) == 0).sum())
    print(f"Test rows with zero known tokens (OOV): {n_oov_rows} / {X_test_bow.shape[0]}")

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=SEED
    )
    mlp.fit(X_train_bow, y_train)
    print("Converged after", mlp.n_iter_, "iterations")

    y_pred_mlp = mlp.predict(X_test_bow)

    print(label)
    print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred_mlp))
    print(classification_report(y_test, y_pred_mlp, zero_division=0))
    return mlp, y_pred_mlp


def train_and_eval_mlp_embeddings(X_train, y_train, X_test, y_test, label):
    """Train an MLP on DistilBERT embeddings and print its results on the test set."""
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=SEED)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print(f"=== {label} ===")
    print("n_iter_:", mlp.n_iter_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return mlp, y_pred


# SVM DistilBERT

def train_and_eval_svm_emb(X_train, y_train, X_test, y_test, label):
    """Train a linear SVM on DistilBERT embeddings and print its results on the test set."""
    # StandardScaler puts all 768 numbers on the same scale, which helps the SVM train
    svm = make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=300, random_state=SEED))
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f"=== {label} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    return svm, y_pred


# Decision Tree

def train_and_eval_dt(X_train, y_train, X_test, y_test, label):
    """Train a decision tree (on BoW or embeddings) and print its results on the test set."""
    # Initialize Decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=SEED)

    # Train model
    dt_classifier.fit(X_train, y_train)

    # Predict on data
    y_pred_dt = dt_classifier.predict(X_test)

    # Print results
    print(f"=== {label} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt, zero_division=0))

    return dt_classifier, y_pred_dt


# Loading saved models (used by evaluate.py and chat.py)

BOW_MODELS = ["rule", "svm", "mlp", "dt"]
EMBEDDING_MODELS = ["svm_emb", "mlp_emb", "dt_emb"]


def load_model(name):
    """Load a model saved by train.py, returns None if it hasn't been trained yet."""
    if name == "rule":
        return {}
    path = os.path.join(MODELS_DIR, name + ".joblib")
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


def predict(name, model, utterances):
    """Predict the dialog acts of a list of (lower cased) utterances with one of the saved models."""
    if name == "rule":
        return [classify(u) for u in utterances]
    if name in EMBEDDING_MODELS:
        from embeddings import get_embeddings # Only load DistilBERT when it is needed
        return list(model["classifier"].predict(get_embeddings(utterances)))
    return list(model["classifier"].predict(model["vectorizer"].transform(utterances)))
