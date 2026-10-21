import os
import re
import io
import sys
import copy
import json
import random
import argparse
import pyglet
from datetime import datetime
from dataclasses import dataclass, field
from gtts import gTTS
import pandas as pd
from dialog_acts import BASE_DIR, clean_utterance
from classifiers import load_model, predict
import slot_extraction
from slot_extraction import SLOTS, MATCHERS, extract, normalize, warm_up
from reasoning import satisfies, explain

CLASSIFIER = "svm"
RECOMMENDATION_MODES = ["eager", "patient"]
RESTAURANT_FILE = os.path.join(BASE_DIR, "data", "restaurant_info.csv")
EXTENDED_FILE = os.path.join(BASE_DIR, "data", "restaurant_info_extended.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")

WELCOME = 1
ASK_MISSING = 2
SUGGEST = 3
NO_MATCH = 4
GIVE_INFO = 5
FINISHED = 6
CONFIRM = 7
ASK_EXTRA = 8

SLOT_ORDER = ["food", "area", "pricerange"] 
RELAX_ORDER = ["pricerange", "area"]  # preferences that may be dropped for a partly matching suggestion, food never is

TEMPLATES = {
    "welcome": "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, "
               "price range or food type. How may I help you?",
    "ask_area": "What part of town do you have in mind?",
    "ask_food": "What kind of food would you like?",
    "ask_pricerange": "Would you like something in the cheap, moderate, or expensive price range?",
    "confirm_value": "I did not recognize '{given}'. Did you mean {suggested}?",
    "unknown_food": "Sorry, I am not familiar with {value} food. What kind of food would you like?",
    "ask_extra": "I found some restaurants matching your preferences. Do you have any additional requirements? "
                 "I can look for a touristic restaurant, a romantic one, one that is suitable for children, "
                 "or one where the waiter assigns you a seat.",
    "which_extra": "Which of those would you like?",
    "extra_not_understood": "I can only take touristic, romantic, children or assigned seats into account.",
    "suggest": "{name} is a nice restaurant{details}.",
    "suggest_alternative": "How about {name}? It is a nice restaurant{details}.",
    "suggest_reason": "I recommend {name}, a nice restaurant{details}; {reasons}.",
    "suggest_alternative_reason": "How about {name}? It is a nice restaurant{details}; {reasons}.",
    "no_match": "I'm sorry, there is no restaurant{constraints}.",
    "no_more_alternatives": "I'm sorry, there is no other restaurant{constraints}.",
    "no_match_extra": "I'm sorry, there is no restaurant{constraints} that matches your additional "
                      "requirement ({requirements}).",
    "no_more_alternatives_extra": "I'm sorry, there is no other restaurant{constraints} that matches your "
                                  "additional requirement ({requirements}).",
    "change_question": " Would you like to change the area, food type or price range?",
    "change_question_extra": " Would you like to change your preferences or drop that requirement?",
    "partial_offer": " The closest match is {name}, a restaurant{details}, but {differences}. "
                     "Would you like that one, or would you like to change your preferences?",
    "partial_next": "How about {name}? It is a restaurant{details}, but {differences}. "
                    "Would you like that one, or would you like to change your preferences?",
    "no_more_partial": "I'm afraid that's all the options I have. Would you like to change a preference?",
    "difference_pricerange": "it is in the {actual} price range instead of {wanted}",
    "difference_area": "it is in the {actual} of town instead of the {wanted}",
    "difference_area_unknown": "I don't know which part of town it is in",
    "accept_offer": "Okay, {name} it is. Can I help you with anything else?",
    "which_change": "Which preference would you like to change: the food type, area or price range?",
    "give_phone": "The phone number of {name} is {phone}.",
    "give_addr": "{name} is at {addr}.",
    "give_postcode": "The postcode of {name} is {postcode}.",
    "give_food": "{name} serves {food}.",
    "give_area": "{name} is in the {area} of town.",
    "give_pricerange": "{name} is in the {pricerange} price range.",
    "unknown_info": "I'm sorry, I don't know the {info} of {name}.",
    "what_info": "I can tell you the phone number, address or postcode of {name}.",
    "answer_yes": "Yes, {fact}.",
    "answer_no": "No, {fact}.",
    "fact_area": "{name} is in the {area} of town",
    "fact_food": "{name} serves {food}",
    "fact_pricerange": "{name} is in the {pricerange} price range",
    "anything_else": "Can I help you with anything else?",
    "no_suggestion_yet": "I haven't suggested a restaurant yet.",
    "not_understood": "Sorry, I didn't understand that.",
    "goodbye": "Goodbye, enjoy your meal!",
}

INFO_NAMES = {"phone": "phone number", "addr": "address", "postcode": "postcode",
              "food": "food type", "area": "area", "pricerange": "price range"}

REQUIREMENT_PATTERNS = {
    "touristic": r"\b(touristic|touristy|tourist|tourists)\b",
    "assigned seats": r"\bassigned (seat|seats|seating)\b|\bassign (me )?a seat\b|\bseated by\b",
    "children": r"\b(children|child|kid|kids|family friendly)\b",
    "romantic": r"\b(romantic|romance|date night)\b",
}

NEGATION_WORDS = {"not", "no", "dont", "doesnt", "without", "non", "never", "nothing"}

NO_EXTRA_PATTERN = (r"\b(no|none|nothing|nope|dont care|do not care|doesnt matter|does not matter|"
                    r"dont mind|no preference|no requirements|thats all|that is all)\b")


def render(key, **values):
    """Fill in a response template."""
    return TEMPLATES[key].format(**values)


def food_text(food):
    """ "spanish" -> "spanish food", but "seafood" stays "seafood"."""
    return food if food.endswith("food") else f"{food} food"


def details(restaurant, leave_out=None):
    """Description of a restaurant for the suggestion, leaving out properties that are unknown."""
    parts = []
    if restaurant["food"]:
        parts.append(f" serving {food_text(restaurant['food'])}")
    if restaurant["area"] and leave_out != "area":
        parts.append(f" in the {restaurant['area']} of town")
    if restaurant["pricerange"] and leave_out != "pricerange":
        parts.append(f" in the {restaurant['pricerange']} price range")
    return "".join(parts)


def constraints(prefs):
    """The user's preferences as text, e.g. " serving spanish food in the north of town"."""
    parts = []
    if prefs["food"] not in (None, "dontcare"):
        parts.append(f" serving {food_text(prefs['food'])}")
    if prefs["area"] not in (None, "dontcare"):
        parts.append(f" in the {prefs['area']} of town")
    if prefs["pricerange"] not in (None, "dontcare"):
        parts.append(f" in the {prefs['pricerange']} price range")
    return "".join(parts)


def requirement_labels(extra):
    """The additional requirements as text, e.g. "romantic, not touristic"."""
    return ", ".join(f"{'' if value else 'not '}{prop}" for prop, value in extra.items())


REQUIRED_COLUMNS = ["restaurantname", "pricerange", "area", "food"]
REASONING_COLUMNS = ["food quality", "crowdedness", "length of stay"]


def _load_restaurants(path=None):
    """
    The restaurant database. By default the extended file is used when it is there, because the
    reasoning component needs its extra columns; the system still runs on the basic file without them.
    Column names are lower cased and underscores become spaces, so that both "food quality" and
    "food_quality" match the column names the inference rules use.
    """
    if path is None:
        path = EXTENDED_FILE if os.path.exists(EXTENDED_FILE) else RESTAURANT_FILE
    df = pd.read_csv(path).fillna("")
    df.columns = [str(c).strip().lower().replace("_", " ") for c in df.columns]
    for column in df.columns:
        if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip().str.lower()
    return df


RESTAURANTS = _load_restaurants()

REASONING_AVAILABLE = all(column in RESTAURANTS.columns for column in REASONING_COLUMNS)


def use_restaurant_file(path):
    """
    Use another restaurant database (--restaurants). It needs at least the columns restaurantname,
    pricerange, area and food; without food quality, crowdedness and length of stay the additional
    requirements are skipped. Raises ValueError with a readable message if the file can't be used.
    """
    global RESTAURANTS, REASONING_AVAILABLE
    if not os.path.isfile(path):
        raise ValueError(f"restaurant file '{path}' not found")
    try:
        df = _load_restaurants(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read '{path}' as a CSV file ({error})")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"'{path}' is missing the column(s): {', '.join(missing)}")
    RESTAURANTS = df
    REASONING_AVAILABLE = all(column in df.columns for column in REASONING_COLUMNS)
    slot_extraction.use_restaurant_file(path)  # So the new food types, areas etc. are recognised


def restaurant_file_option(parser):
    """The --restaurants option, shared with run_reference_dialogs.py."""
    parser.add_argument("--restaurants", metavar="CSV",
                        help="restaurant database to use (default: data/restaurant_info_extended.csv)")


def apply_restaurant_file_option(args):
    """Load the file given with --restaurants, or stop with an error message."""
    if args.restaurants:
        try:
            use_restaurant_file(args.restaurants)
        except ValueError as error:
            print(f"Error: {error}")
            sys.exit(1)


def lookup(prefs, rng, extra=None):
    """
    All restaurants matching the preferences in random order, "dontcare" matches everything.
    When additional requirements are given, the reasoning component filters the matches further.
    """
    df = RESTAURANTS
    for slot in SLOTS:
        if prefs[slot] not in (None, "dontcare"):
            df = df[df[slot] == prefs[slot]]
    restaurants = df.to_dict("records")
    if extra:
        restaurants = [r for r in restaurants if satisfies(r, extra)]
    rng.shuffle(restaurants)
    return restaurants


def split_requirements(utterance):
    """
    The derived properties the user asks for, e.g. {"romantic": True} or {"children": False},
    and the rest of the utterance. Keyword matching like the slot extraction, with the words
    before the keyword deciding the value.
    The keywords are removed from the rest, because the slot extraction would otherwise send
    them to its fallback matcher: "romantic" is only two edits away from the food type
    "romanian", so "i want a romantic place" would be read as a request for romanian food.
    """
    text = " ".join(normalize(utterance))
    found, spans = {}, []
    for prop, pattern in REQUIREMENT_PATTERNS.items():
        match = re.search(pattern, text)
        if match is None:
            continue
        before = text[:match.start()].split()[-3:]
        found[prop] = not any(word in NEGATION_WORDS for word in before)
        spans.append(match.span())
    rest = text
    for start, end in sorted(spans, reverse=True):
        rest = rest[:start] + rest[end:]
    return found, re.sub(r"\s+", " ", rest).strip()


@dataclass
class DialogState:
    state: int = WELCOME
    prefs: dict = field(default_factory=lambda: {slot: None for slot in SLOTS})
    pending: dict = field(default_factory=dict)
    extra: dict = None
    asked_slot: str = None
    candidates: list = None
    current: dict = None
    last_utterance: str = ""
    matcher: str = "levenshtein"
    transparency: bool = True
    tts: bool = False
    recommendation_mode: str = "patient"
    unfamiliar: str = None                      # food word the user asked for that isn't in the database
    seen: list = field(default_factory=list)    # names of all restaurants suggested or offered in this dialog
    offered: dict = None                        # partly matching restaurant offered after a failed search
    partial: list = None                        # the other partly matching restaurants, offered on request
    rng: random.Random = field(default_factory=random.Random)

def show(state, reply):
    """Print a system utterance, then read it aloud if TTS is enabled."""
    print("system:", reply)
    if state.tts:
        tts_speak(reply)

def start(matcher="levenshtein", seed=None, transparency=True, recommendation_mode="patient", tts=False):
    """A new dialog, returns the start state and the welcome message."""
    state = DialogState(matcher=matcher,
                        transparency=transparency, 
                        recommendation_mode=recommendation_mode, 
                        rng=random.Random(seed), 
                        tts=tts)
    return state, _say(state, render("welcome"))


def _say(state, utterance):
    state.last_utterance = utterance
    return utterance

def _reprompt(state, prefix):
    """Say something before repeating the last question, without changing what "repeat" repeats."""
    return f"{prefix} {state.last_utterance}"


def _all_info_known(state):
    """Decision node "All information known?": ask for the first missing preference or look up restaurants."""
    # in patient mode, the system waits until all preferences are known before suggesting a restaurant, in eager mode it suggests as soon as it has at least one preference
    if state.recommendation_mode == "eager" and any(value is not None for value in state.prefs.values()):
        return _additional_requirements_known(state)

    for slot in SLOT_ORDER:
        if state.prefs[slot] is None:
            state.state = ASK_MISSING
            state.asked_slot = slot
            return _say(state, render("ask_" + slot))
    return _additional_requirements_known(state)


def _additional_requirements_known(state):
    """
    Decision node "Additional requirements known?": all preferences are in, so the system asks
    once whether the user wants one of the derived properties before it recommends a restaurant.
    """
    #if in eager mode, don't wait for the user to answer the question about additional requirements, just go ahead and look for a restaurant
    if state.recommendation_mode == "eager":
        return _combination_available(state)

    # Additional requirements only help to choose between several restaurants, so with 0 or 1
    # matches, or once a restaurant was recommended, the system answers directly
    if state.extra is None and REASONING_AVAILABLE and state.current is None:
        if len(lookup(state.prefs, state.rng)) >= 2:
            state.state = ASK_EXTRA
            state.asked_slot = None
            return _say(state, render("ask_extra"))
    return _combination_available(state)


def _combination_available(state, alternative=False):
    """Decision node "Is there a combination available?": suggest a restaurant or say there is none."""
    state.asked_slot = None
    if state.candidates is None:
        state.candidates = lookup(state.prefs, state.rng, state.extra)
    if state.candidates:
        state.current = state.candidates.pop(0)
        state.seen.append(state.current["restaurantname"])
        state.state = SUGGEST
        return _say(state, _suggestion(state, alternative))

    # No (more) exact matches: say so, and offer the closest partly matching restaurant if there is one
    state.state = NO_MATCH
    if state.extra:
        key = "no_more_alternatives_extra" if alternative else "no_match_extra"
        message = render(key, constraints=constraints(state.prefs), requirements=requirement_labels(state.extra))
    else:
        key = "no_more_alternatives" if alternative else "no_match"
        message = render(key, constraints=constraints(state.prefs))
    state.partial = _partial_matches(state)
    if state.partial:
        return _say(state, message + _offer_partial(state, "partial_offer"))
    return _say(state, message + render("change_question_extra" if state.extra else "change_question"))


def _partial_matches(state):
    """
    Restaurants that match all preferences except one, for when nothing (more) matches exactly.
    First the price range is dropped, then the area; the food type is always kept. A preference is
    only dropped if another real preference remains, so the suggestion is still similar.
    Restaurants that were already suggested or offered are skipped.
    """
    specified = [slot for slot in SLOTS if state.prefs[slot] not in (None, "dontcare")]
    if len(specified) < 2:
        return []
    matches, names = [], set(state.seen)
    for slot in RELAX_ORDER:
        if slot not in specified:
            continue
        relaxed = {**state.prefs, slot: "dontcare"}
        for restaurant in lookup(relaxed, state.rng, state.extra):
            if restaurant["restaurantname"] not in names:
                names.add(restaurant["restaurantname"])
                matches.append((restaurant, slot))
    return matches


def _offer_partial(state, key):
    """Offer the next partly matching restaurant and say how it differs from the preferences."""
    restaurant, slot = state.partial.pop(0)
    state.offered = restaurant
    state.seen.append(restaurant["restaurantname"])
    if not restaurant[slot]:
        difference = render("difference_area_unknown")
    else:
        difference = render("difference_" + slot, actual=restaurant[slot], wanted=state.prefs[slot])
    return render(key, name=restaurant["restaurantname"], details=details(restaurant, leave_out=slot), differences=difference)


def _suggestion(state, alternative):
    """
    The recommendation itself. With the transparency feature on, the inference chain behind the
    additional requirement is given as well; with it off, only the restaurant is named.
    """
    name, detail = state.current["restaurantname"], details(state.current)
    if state.extra and state.transparency:
        key = "suggest_alternative_reason" if alternative else "suggest_reason"
        reasons = explain(state.current, state.extra)
        return render(key, name=name, details=detail, reasons=reasons)
    key = "suggest_alternative" if alternative else "suggest"
    return render(key, name=name, details=detail)


def _update_preferences(state, extraction):
    """Store the extracted preferences, returns True if anything changed."""
    changed = False
    for slot, value in extraction.negated.items():
        if state.prefs[slot] == value:
            state.prefs[slot] = None
            changed = True
    for slot, value in extraction.slots.items():
        state.pending.pop(slot, None)
        if state.prefs[slot] != value:
            state.prefs[slot] = value
            changed = True
    for slot, guess in extraction.uncertain.items():
        state.pending[slot] = guess
        changed = True
    if "food" in extraction.unknown and "food" not in extraction.slots:
        # a food type that isn't in the database ("swedish"): not stored, the user is asked again
        state.unfamiliar = extraction.unknown["food"]
        state.pending.pop("food", None)
        state.prefs["food"] = None
        changed = True
    if changed:
        state.candidates = None
        state.offered = state.partial = None
    return changed


def _update_requirements(state, requirements, replace=False):
    """
    Store additional requirements found in the utterance, returns True if there were any.
    New requirements are added to the ones already given, except after a failed search or an
    "how about ..." question, where the user is replacing the requirement rather than adding one.
    """
    if not requirements:
        return False
    state.extra = requirements if replace else {**(state.extra or {}), **requirements}
    state.candidates = None
    state.offered = state.partial = None
    return True


def _after_update(state):
    """
    Ask again for an unfamiliar food type, then confirm uncertain values, otherwise go to the
    decision node "All information known?".
    """
    if state.unfamiliar:
        word, state.unfamiliar = state.unfamiliar, None
        state.state = ASK_MISSING
        state.asked_slot = "food"
        return _say(state, render("unknown_food", value=word))
    if state.pending:
        slot, (given, suggested) = next(iter(state.pending.items()))
        state.state = CONFIRM
        state.asked_slot = slot
        return _say(state, render("confirm_value", given=given, suggested=suggested))
    return _all_info_known(state)


def _collecting(state, act, utterance):
    """States 1, 2 and 4: the system is gathering preferences."""
    requirements, rest = split_requirements(utterance)
    extraction = extract(rest, state.asked_slot, state.matcher)
    # Preferences and requirements in the same utterance are both kept
    prefs_changed = _update_preferences(state, extraction)
    extra_changed = _update_requirements(state, requirements, replace=state.state == NO_MATCH)
    if prefs_changed:
        return _after_update(state)
    if extra_changed:
        return _combination_available(state) if state.state == NO_MATCH else _after_update(state)
    if state.state == NO_MATCH:
        reply = _after_no_match(state, act, extraction)
        if reply:
            return reply
    if act == "request" and state.current and extraction.requests:
        return _give_info(state, extraction.requests)
    if act == "request":
        return _reprompt(state, render("no_suggestion_yet"))
    if state.state == WELCOME and act in ("inform", "hello", "null"):
        return _all_info_known(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return state.last_utterance


def _after_no_match(state, act, extraction):
    """
    State 4 without new preferences: the user reacts to a partly matching restaurant that was
    offered, or asks for another one. Returns None if the utterance is not about that.
    """
    wants_info = extraction.requests and (act == "request" or not extraction.found_preference())
    if state.offered and (act in ("affirm", "ack") or wants_info):
        # Accepting the offer, asking for its phone number etc. also counts as accepting it
        state.current, state.offered = state.offered, None
        state.candidates = []
        if wants_info:
            return _give_info(state, extraction.requests)
        state.state = SUGGEST
        return _say(state, render("accept_offer", name=state.current["restaurantname"]))
    if act in ("reqalts", "reqmore"):
        if state.partial:
            return _say(state, _offer_partial(state, "partial_next").strip())
        state.offered = None
        return _say(state, render("no_more_partial"))
    if state.offered and act in ("negate", "deny"):
        state.offered = None
        return _say(state, render("which_change"))
    return None


def _confirm(state, act, utterance):
    """State 7: the user confirms or rejects a value found by Levenshtein or semantic similarity."""
    slot, (_, suggested) = next(iter(state.pending.items()))
    requirements, rest = split_requirements(utterance)
    _update_requirements(state, requirements)
    extraction = extract(rest, None, state.matcher)
    if act == "affirm":
        del state.pending[slot]
        state.prefs[slot] = suggested
        state.candidates = None
        _update_preferences(state, extraction)
        return _after_update(state)
    if act in ("negate", "deny", "inform") or extraction.found_preference():
        del state.pending[slot]
        _update_preferences(state, extraction)
        return _after_update(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return state.last_utterance


def _ask_extra(state, act, utterance):
    """
    State 8: the system has all preferences and asks for additional requirements, which the
    reasoning component then derives for every candidate restaurant.
    """
    requirements, rest = split_requirements(utterance)
    if _update_requirements(state, requirements):
        return _combination_available(state)
    extraction = extract(rest, None, state.matcher)
    if _update_preferences(state, extraction):
        return _after_update(state)
    if act in ("negate", "deny") or re.search(NO_EXTRA_PATTERN, " ".join(normalize(utterance))):
        state.extra = {}
        return _combination_available(state)
    if act == "affirm":
        return _reprompt(state, render("which_extra"))
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return _reprompt(state, render("extra_not_understood"))


def _value_text(slot, value):
    return food_text(value) if slot == "food" else value


def _give_info(state, requests):
    """Answer requests about the suggested restaurant (state 5)."""
    name = state.current["restaurantname"]
    answers = []
    for info in requests:
        if state.current.get(info):
            answers.append(render("give_" + info, name=name, **{info: _value_text(info, state.current[info])}))
        else:
            answers.append(render("unknown_info", info=INFO_NAMES[info], name=name))
    state.state = GIVE_INFO
    return _say(state, " ".join(answers))


def _answer_confirm(state, extraction):
    """Answer a yes/no question about the suggested restaurant, e.g. "is it in the centre?"."""
    for slot, value in extraction.slots.items():
        if value == "dontcare":
            continue
        fact = render("fact_" + slot, name=state.current["restaurantname"],
                      **{slot: _value_text(slot, state.current[slot])})
        key = "answer_yes" if state.current[slot] == value else "answer_no"
        return _say(state, render(key, fact=fact))
    return None


def _suggested(state, act, utterance):
    """States 3 and 5: a restaurant was suggested, the user can ask about it or for another one."""
    requirements, rest = split_requirements(utterance)
    extraction = extract(rest, None, state.matcher)
    if extraction.requests and (act == "request" or not extraction.found_preference()):
        return _give_info(state, extraction.requests)
    if act == "request":
        return _say(state, render("what_info", name=state.current["restaurantname"]))
    if act == "confirm":
        answer = _answer_confirm(state, extraction)
        if answer:
            return answer
    if act in ("reqalts", "reqmore"):
        if _update_preferences(state, extraction):
            return _after_update(state)
        if _update_requirements(state, requirements, replace=True):
            return _combination_available(state)
        return _combination_available(state, alternative=True)
    if act in ("inform", "negate", "deny"):
        if _update_preferences(state, extraction):
            return _after_update(state)
        if _update_requirements(state, requirements):
            return _combination_available(state)
    if act == "null":
        return _reprompt(state, render("not_understood"))
    return _say(state, render("anything_else"))


HANDLERS = {WELCOME: _collecting, ASK_MISSING: _collecting, NO_MATCH: _collecting,
            SUGGEST: _suggested, GIVE_INFO: _suggested, CONFIRM: _confirm, ASK_EXTRA: _ask_extra}


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
        return start(state.matcher, state.rng.random(), state.transparency, state.recommendation_mode)
    if act == "repeat":
        return state, state.last_utterance
    return state, HANDLERS[state.state](state, act, utterance)


def start_log(config, log_dir=LOG_DIR):
    """Open a new log file and write the configuration of this dialog, returns the path."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"dialog_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    log(path, {"event": "config", **config})
    return path


def log(path, record):
    """Append one record to the log file; logging never interrupts the dialog."""
    if path is None:
        return
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as error:
        print(f"(warning: could not write to the log file: {error})")

def tts_speak(utterance):
    """Speak the text using TTS (text-to-speech)."""
    with io.BytesIO() as f:
        gTTS(text=utterance, lang='en').write_to_fp(f)
        f.seek(0)

        player = pyglet.media.load('_.mp3', file=f).play()
        while player.playing:
            pyglet.app.platform_event_loop.dispatch_posted_events()
            pyglet.clock.tick()
  
def main():
    parser = argparse.ArgumentParser(description="Restaurant recommendation dialog system.")
    parser.add_argument("--matcher", default="levenshtein", choices=MATCHERS,
                        help="fallback for slot extraction when keyword matching fails (default: levenshtein)")
    parser.add_argument("--transparency", default="on", choices=["on", "off"],
                        help="show the reasoning behind a recommendation or only the restaurant (default: on)")
    parser.add_argument("--debug", action="store_true", help="show the dialog act, state and preferences every turn")
    parser.add_argument("--seed", type=int, default=None, help="random seed for choosing between restaurants")
    parser.add_argument("--no-log", action="store_true", help="do not write an interaction log")
    parser.add_argument("--log-dir", default=LOG_DIR, help=f"where to write interaction logs (default: {LOG_DIR})")
    parser.add_argument("--recommendation-mode", default="patient", choices=RECOMMENDATION_MODES, help="whether to suggest a restaurant as soon as one preference is known (eager) or wait until all preferences are known (patient, default)")
    parser.add_argument("--tts", action="store_true", help="enable TTS (text-to-speech) for the system utterances")
    restaurant_file_option(parser)
    args = parser.parse_args()
    apply_restaurant_file_option(args)
    model = load_model(CLASSIFIER)
    if model is None:
        print(f"Error: model '{CLASSIFIER}' not found, run python train.py first")
        sys.exit(1)
    if args.matcher == "semantic":
        print("Loading DistilBERT for the semantic matcher...")
        warm_up()
    if not REASONING_AVAILABLE:
        print("Note: the restaurant file has no food quality, crowdedness and length of stay columns, "
              "so additional requirements are skipped.")
    transparency = args.transparency == "on"
    log_path = None if args.no_log else start_log(
        {"classifier": CLASSIFIER, "matcher": args.matcher, "transparency": transparency,
         "seed": args.seed, "reasoning": REASONING_AVAILABLE, "recommendation_mode": args.recommendation_mode,
         "restaurants": args.restaurants or "default"}, args.log_dir)
    state, reply = start(args.matcher, args.seed, transparency, args.recommendation_mode, tts=args.tts)
    show(state, reply)
    log(log_path, {"event": "system", "turn": 0, "state": state.state, "utterance": reply})
    turn = 0
    while state.state != FINISHED:
        try:
            text = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        utterance = clean_utterance(text)
        if utterance == "":
            show(state, "Please type a sentence.")
            continue
        turn += 1
        act = predict(CLASSIFIER, model, [utterance])[0]
        previous = state.state
        state, reply = transition(state, act, text)
        if args.debug:
            pending = f", pending={state.pending}" if state.pending else ""
            print(f"  [act={act}, state {previous} -> {state.state}, prefs={state.prefs}, "
                  f"extra={state.extra}{pending}]")
        show(state, reply) 
        log(log_path, {"event": "turn", "turn": turn, "user": text, "act": act,
                       "state_before": previous, "state_after": state.state,
                       "prefs": state.prefs, "extra": state.extra,
                       "suggested": state.current["restaurantname"] if state.current else None,
                       "system": reply})
    log(log_path, {"event": "end", "turns": turn, "state": state.state,
                   "prefs": state.prefs, "extra": state.extra,
                   "suggested": state.current["restaurantname"] if state.current else None})
    if log_path:
        print(f"(interaction log: {log_path})")


if __name__ == "__main__":
    main()