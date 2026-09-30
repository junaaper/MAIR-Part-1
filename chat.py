# Interactive prompt: type an utterance and see the predicted dialog act, until you type quit
import sys
import argparse

from dialog_acts import clean_utterance
from classifiers import BOW_MODELS, load_model, predict


def main():
    parser = argparse.ArgumentParser(description="Classify utterances typed by the user.")
    parser.add_argument("--model", default="svm", choices=BOW_MODELS,
                        help="classifier to start with (default: svm)")
    args = parser.parse_args()

    # Load all models once so switching is instant
    models = {}
    for name in BOW_MODELS:
        models[name] = load_model(name)
        if models[name] is None:
            print(f"Error: model '{name}' not found, run python train.py first")
            sys.exit(1)

    current = args.model
    help_text = ("Type a sentence to see its dialog act.\n"
                 ":model <name> switches classifier (" + ", ".join(BOW_MODELS) + ")\n"
                 ":models lists the classifiers, :help shows this again, quit or exit stops")
    print(help_text)
    print(f"\nUsing classifier: {current}")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text == "":
            continue

        if text.lower() in ("quit", "exit"):
            break

        if text.startswith(":"):
            command = text.lower().split()
            if command[0] == ":model" and len(command) == 2 and command[1] in BOW_MODELS:
                current = command[1]
                print(f"Switched to classifier: {current}")
            elif command[0] == ":model":
                print("Unknown classifier, choose one of:", ", ".join(BOW_MODELS))
            elif command[0] == ":models":
                print("Available classifiers:", ", ".join(BOW_MODELS), f"(current: {current})")
            else:
                print(help_text)
            continue

        utterance = clean_utterance(text) # Also converts the input to lower case
        if utterance == "":
            print("No words found in the input, please type a sentence.")
            continue

        dialog_act = predict(current, models[current], [utterance])[0]

        # If none of the words are in the training vocabulary the BoW vector is empty
        note = ""
        if current != "rule" and models[current]["vectorizer"].transform([utterance]).sum() == 0:
            note = "  (no known words, prediction is unreliable)"

        print(f"dialog act: {dialog_act}  (classifier: {current}){note}")

    print("Goodbye!")


if __name__ == "__main__":
    main()
