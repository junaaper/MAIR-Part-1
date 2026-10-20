# Replay the user turns of reference_dialogs.txt (or another dialog file) through the dialog system
# and print the expected and actual system utterances side by side
# Usage: python run_reference_dialogs.py [--file PATH] [--restaurants CSV] [--dialog N ...]
#                                        [--recommendation-mode patient|eager] [--seed N]
import os
import re
import sys
import argparse

from dialog_acts import BASE_DIR, clean_utterance
from classifiers import load_model, predict
import dialog_manager as dm

REFERENCE_FILE = os.path.join(BASE_DIR, "reference_dialogs.txt")


def load_reference_dialogs(path=REFERENCE_FILE):
    """Returns {dialog number: [(speaker, text), ...]}, indented lines continue the previous turn."""
    dialogs, current = {}, None
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip()
            header = re.match(r"--- Dialog (\d+) ---", line)
            if header:
                current = dialogs.setdefault(int(header.group(1)), [])
            elif current is not None and re.match(r"[SU]: ", line):
                current.append((line[0], line[3:].strip()))
            elif current is not None and current and line.startswith(" ") and line.strip():
                speaker, text = current[-1]
                current[-1] = (speaker, f"{text} {line.strip()}")
    return dialogs


def replay(turns, model, args):
    """Run the user turns through the system, returns a list of printable lines."""
    options = {"recommendation_mode": args.recommendation_mode} if hasattr(dm, "RECOMMENDATION_MODES") else {}
    state, reply = dm.start(args.matcher, args.seed, transparency=True, **options)
    expected_system = [text for speaker, text in turns if speaker == "S"]
    user_turns = [text for speaker, text in turns if speaker == "U"]
    lines = ([f"  S(exp): {expected_system[0]}"] if expected_system else []) + [f"  S(got): {reply}"]

    for i, text in enumerate(user_turns, start=1):
        act = predict(dm.CLASSIFIER, model, [clean_utterance(text)])[0]
        previous = state.state
        state, reply = dm.transition(state, act, text)
        lines.append(f"  U:      {text}")
        lines.append(f"          [act={act}, state {previous}->{state.state}, prefs={state.prefs}, extra={state.extra}]")
        if i < len(expected_system):
            lines.append(f"  S(exp): {expected_system[i]}")
        lines.append(f"  S(got): {reply}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Replay the reference dialogs through the dialog system.")
    parser.add_argument("--file", default=REFERENCE_FILE,
                        help="dialog file in the reference format (default: reference_dialogs.txt)")
    parser.add_argument("--dialog", type=int, nargs="*", help="only these dialog numbers (default: all)")
    parser.add_argument("--recommendation-mode", default="patient", choices=["eager", "patient"],
                        help="only used if the dialog manager has eager/patient modes")
    parser.add_argument("--matcher", default="levenshtein", choices=["levenshtein", "semantic"])
    parser.add_argument("--seed", type=int, default=1)
    dm.restaurant_file_option(parser)
    args = parser.parse_args()
    dm.apply_restaurant_file_option(args)

    if not os.path.isfile(args.file):
        print(f"Error: dialog file '{args.file}' not found")
        sys.exit(1)
    dialogs = load_reference_dialogs(args.file)
    if not dialogs:
        print(f"Error: no dialogs found in '{args.file}' (expected '--- Dialog N ---' headers and 'S: ' / 'U: ' lines)")
        sys.exit(1)
    unknown = [n for n in args.dialog or [] if n not in dialogs]
    if unknown:
        print(f"Error: dialog(s) {', '.join(map(str, unknown))} not in '{args.file}' "
              f"(it has {min(dialogs)}-{max(dialogs)})")
        sys.exit(1)

    model = load_model(dm.CLASSIFIER)
    if model is None:
        print(f"Error: model '{dm.CLASSIFIER}' not found, run python train.py first")
        sys.exit(1)
    for number in args.dialog or sorted(dialogs):
        print(f"\n===== Dialog {number} =====")
        print("\n".join(replay(dialogs[number], model, args)))


if __name__ == "__main__":
    main()
