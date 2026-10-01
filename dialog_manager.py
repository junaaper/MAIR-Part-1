# Dialog manager for restaurant recommendation via a state transition function 
# Usage: python dialog_manager.py [--matcher levenshtein|semantic] [--debug] [--seed N]
import os
import sys
import copy
import random
import argparse
from dataclasses import dataclass, field

import pandas as pd

from dialog_acts import BASE_DIR, clean_utterance
from classifiers import load_model, predict
from slot_extraction import SLOTS, MATCHERS, extract, warm_up

CLASSIFIER = "svm"  # Chosen due to our results from Part 1a: best accuracy on the grouped split, tied best accuracy for original
RESTAURANT_FILE = os.path.join(BASE_DIR, "data", "restaurant_info.csv")

# States, numbered according to our state transition diagram
WELCOME = 1
ASK_MISSING = 2
SUGGEST = 3
NO_MATCH = 4
GIVE_INFO = 5
FINISHED = 6
CONFIRM = 7

SLOT_ORDER = ["area", "food", "pricerange"]  # The order in which missing preferences are asked (as in DSTC2)


# Response templates

TEMPLATES = {
    "welcome": "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, "
               "price range or food type. How may I help you?",
    "ask_area": "What part of town do you have in mind?",
    "ask_food": "What kind of food would you like?",
    "ask_pricerange": "Would you like something in the cheap, moderate, or expensive price range?",
    "confirm_value": "I did not recognize '{given}'. Did you mean {suggested}?",
    "suggest": "{name} is a nice restaurant{details}.",
    "suggest_alternative": "How about {name}? It is a nice restaurant{details}.",
    "no_match": "I'm sorry, there is no restaurant{constraints}. "
                "Would you like to change the area, food type or price range?",
    "no_more_alternatives": "I'm sorry, there is no other restaurant{constraints}. "
                            "Would you like to change the area, food type or price range?",
    "give_phone": "The phone number of {name} is {phone}.",
    "give_addr": "{name} is at {addr}.",
    "give_postcode": "The postcode of {name} is {postcode}.",
    "give_food": "{name} serves {food} food.",
    "give_area": "{name} is in the {area} of town.",
    "give_pricerange": "{name} is in the {pricerange} price range.",
    "unknown_info": "I'm sorry, I don't know the {info} of {name}.",
    "what_info": "I can tell you the phone number, address or postcode of {name}.",
    "answer_yes": "Yes, {fact}.",
    "answer_no": "No, {fact}.",
    "fact_area": "{name} is in the {area} of town",
    "fact_food": "{name} serves {food} food",
    "fact_pricerange": "{name} is in the {pricerange} price range",
    "anything_else": "Can I help you with anything else?",
    "no_suggestion_yet": "I haven't suggested a restaurant yet.",
    "not_understood": "Sorry, I didn't understand that.",
    "goodbye": "Goodbye, enjoy your meal!",
}

INFO_NAMES = {"phone": "phone number", "addr": "address", "postcode": "postcode",
              "food": "food type", "area": "area", "pricerange": "price range"}


def render(key, **values):
    """Fill in a response template."""
    return TEMPLATES[key].format(**values)


def details(restaurant):
    """Description of a restaurant for the suggestion, leaving out properties that are unknown."""
    parts = []
    if restaurant["food"]:
        parts.append(f" serving {restaurant['food']} food")
    if restaurant["area"]:
        parts.append(f" in the {restaurant['area']} of town")
    if restaurant["pricerange"]:
        parts.append(f" in the {restaurant['pricerange']} price range")
    return "".join(parts)


def constraints(prefs):
    """The user's preferences as text, e.g. " serving spanish food in the north of town"."""
    parts = []
    if prefs["food"] not in (None, "dontcare"):
        parts.append(f" serving {prefs['food']} food")
    if prefs["area"] not in (None, "dontcare"):
        parts.append(f" in the {prefs['area']} of town")
    if prefs["pricerange"] not in (None, "dontcare"):
        parts.append(f" in the {prefs['pricerange']} price range")
    return "".join(parts)


# Restaurant lookup (minimal version as of now, needs to be replaced by the lookup/reasoning module when we make it)

RESTAURANTS = pd.read_csv(RESTAURANT_FILE).fillna("")


def lookup(prefs, rng):
    """All restaurants matching the preferences in random order, "dontcare" matches everything."""
    df = RESTAURANTS
    for slot in SLOTS:
        if prefs[slot] not in (None, "dontcare"):
            df = df[df[slot] == prefs[slot]]
    restaurants = df.to_dict("records")
    rng.shuffle(restaurants)
    return restaurants


# Dialog state

@dataclass
class DialogState:
    state: int = WELCOME
    prefs: dict = field(default_factory=lambda: {slot: None for slot in SLOTS})
    pending: dict = field(default_factory=dict)  # slot -> (given word, suggested value) waiting for confirmation
    asked_slot: str = None                        # the preference the system just asked for
    candidates: list = None                       # matching restaurants not suggested yet, None = not looked up
    current: dict = None                          # the restaurant that was suggested last
    last_utterance: str = ""                      # the last system utterance, for "repeat"
    matcher: str = "levenshtein"                  # fallback for slot extraction
    rng: random.Random = field(default_factory=random.Random)


def start(matcher="levenshtein", seed=None):
    """A new dialog, returns the start state and the welcome message."""
    state = DialogState(matcher=matcher, rng=random.Random(seed))
    return state, _say(state, render("welcome"))


def _say(state, utterance):
    state.last_utterance = utterance
    return utterance


def _reprompt(state, prefix):
    """Say something before repeating the last question, without changing what "repeat" repeats."""
    return f"{prefix} {state.last_utterance}"


# Diagram decision nodes

def _all_info_known(state):
    """Decision node "All information known?": ask for the first missing preference or look up restaurants."""
    for slot in SLOT_ORDER:
        if state.prefs[slot] is None:
            state.state = ASK_MISSING
            state.asked_slot = slot
            return _say(state, render("ask_" + slot))
    return _combination_available(state)


def _combination_available(state, alternative=False):
    """Decision node "Is there a combination available?": suggest a restaurant or say there is none."""
    state.asked_slot = None
    if state.candidates is None:
        state.candidates = lookup(state.prefs, state.rng)
    if state.candidates:
        state.current = state.candidates.pop(0)
        state.state = SUGGEST
        key = "suggest_alternative" if alternative else "suggest"
        return _say(state, render(key, name=state.current["restaurantname"], details=details(state.current)))
    state.state = NO_MATCH
    key = "no_more_alternatives" if alternative else "no_match"
    return _say(state, render(key, constraints=constraints(state.prefs)))


# Updating the preferences

def _update_preferences(state, extraction):
    """Store the extracted preferences, returns True if anything changed."""
    changed = False
    for slot, value in extraction.negated.items():
        if state.prefs[slot] == value:  # "not chinese" after chinese was chosen: ask again
            state.prefs[slot] = None
            changed = True
    for slot, value in {**extraction.unknown, **extraction.slots}.items():
        state.pending.pop(slot, None)
        if state.prefs[slot] != value:
            state.prefs[slot] = value
            changed = True
    for slot, guess in extraction.uncertain.items():
        state.pending[slot] = guess
        changed = True
    if changed:
        state.candidates = None  # Preferences changed, so the old lookup is no longer valid
    return changed


def _after_update(state):
    """Confirm uncertain values first, otherwise go to the decision node "All information known?"."""
    if state.pending:
        slot, (given, suggested) = next(iter(state.pending.items()))
        state.state = CONFIRM
        state.asked_slot = slot
        return _say(state, render("confirm_value", given=given, suggested=suggested))
    return _all_info_known(state)


# State handlers, one per state in the diagram

def _collecting(state, act, utterance):
    """States 1, 2 and 4: the system is gathering preferences."""
    extraction = extract(utterance, state.asked_slot, state.matcher)

    # Any utterance with preferences is used, so a misclassified "chinese food" still works
    if _update_preferences(state, extraction):
        return _after_update(state)
    if act == "request" and state.current and extraction.requests:
        return _give_info(state, extraction.requests)  # About the last suggestion, e.g. after "no other restaurant"
    if act == "request":
        return _reprompt(state, render("no_suggestion_yet"))
    if state.state == WELCOME and act in ("inform", "hello", "null"):
        return _all_info_known(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return state.last_utterance


def _confirm(state, act, utterance):
    """State 7: the user confirms or rejects a value found by Levenshtein or semantic similarity."""
    slot, (_, suggested) = next(iter(state.pending.items()))
    extraction = extract(utterance, None, state.matcher)

    if act == "affirm":
        del state.pending[slot]
        state.prefs[slot] = suggested
        state.candidates = None
        _update_preferences(state, extraction)  # "yes, and in the north"
        return _after_update(state)
    if act in ("negate", "deny", "inform") or extraction.found_preference():
        del state.pending[slot]
        _update_preferences(state, extraction)  # "no, i meant italian"
        return _after_update(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return state.last_utterance


def _give_info(state, requests):
    """Answer requests about the suggested restaurant (state 5)."""
    name = state.current["restaurantname"]
    answers = []
    for info in requests:
        if state.current.get(info):
            answers.append(render("give_" + info, name=name, **{info: state.current[info]}))
        else:
            answers.append(render("unknown_info", info=INFO_NAMES[info], name=name))
    state.state = GIVE_INFO
    return _say(state, " ".join(answers))


def _answer_confirm(state, extraction):
    """Answer a yes/no question about the suggested restaurant, e.g. "is it in the centre?"."""
    for slot, value in extraction.slots.items():
        if value == "dontcare":
            continue
        fact = render("fact_" + slot, name=state.current["restaurantname"], **{slot: state.current[slot]})
        key = "answer_yes" if state.current[slot] == value else "answer_no"
        return _say(state, render(key, fact=fact))
    return None


def _suggested(state, act, utterance):
    """States 3 and 5: a restaurant was suggested, the user can ask about it or for another one."""
    extraction = extract(utterance, None, state.matcher)

    if extraction.requests and (act == "request" or not extraction.found_preference()):
        return _give_info(state, extraction.requests)
    if act == "request":
        return _say(state, render("what_info", name=state.current["restaurantname"]))
    if act == "confirm":
        answer = _answer_confirm(state, extraction)
        if answer:
            return answer
    if act in ("reqalts", "reqmore"):
        # "how about chinese food" changes the preferences, "anything else" asks for the next restaurant
        if _update_preferences(state, extraction):
            return _after_update(state)
        return _combination_available(state, alternative=True)
    if act in ("inform", "negate", "deny") and _update_preferences(state, extraction):
        return _after_update(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return _say(state, render("anything_else"))


HANDLERS = {WELCOME: _collecting, ASK_MISSING: _collecting, NO_MATCH: _collecting,
            SUGGEST: _suggested, GIVE_INFO: _suggested, CONFIRM: _confirm}


# State transition function

def transition(state, act, utterance):
    """
    The state transition function: takes the current dialog state and the classified user utterance
    (dialog act + text) and returns the next dialog state and the system utterance.
    The input state is not changed.
    """
    state = copy.deepcopy(state)

    if state.state == FINISHED:
        return state, state.last_utterance
    if act in ("bye", "thankyou"):
        state.state = FINISHED
        return state, _say(state, render("goodbye"))
    if act == "restart":
        return start(state.matcher, state.rng.random())
    if act == "repeat":
        return state, state.last_utterance

    return state, HANDLERS[state.state](state, act, utterance)


# Terminal interface

def main():
    parser = argparse.ArgumentParser(description="Restaurant recommendation dialog system.")
    parser.add_argument("--matcher", default="levenshtein", choices=MATCHERS,
                        help="fallback for slot extraction when keyword matching fails (default: levenshtein)")
    parser.add_argument("--debug", action="store_true", help="show the dialog act, state and preferences every turn")
    parser.add_argument("--seed", type=int, default=None, help="random seed for choosing between restaurants")
    args = parser.parse_args()

    model = load_model(CLASSIFIER)
    if model is None:
        print(f"Error: model '{CLASSIFIER}' not found, run python train.py first")
        sys.exit(1)
    if args.matcher == "semantic":
        print("Loading DistilBERT for the semantic matcher...")
        warm_up()

    state, reply = start(args.matcher, args.seed)
    print("system:", reply)

    while state.state != FINISHED:
        try:
            text = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        utterance = clean_utterance(text)
        if utterance == "":
            print("system: Please type a sentence.")
            continue

        act = predict(CLASSIFIER, model, [utterance])[0]
        previous = state.state
        state, reply = transition(state, act, text)

        if args.debug:
            pending = f", pending={state.pending}" if state.pending else ""
            print(f"  [act={act}, state {previous} -> {state.state}, prefs={state.prefs}{pending}]")
        print("system:", reply)


if __name__ == "__main__":
    main()
