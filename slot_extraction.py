# Slot extraction: find the user's preferences (area, food, pricerange) and requests in an utterance
# Keyword matching comes first, Levenshtein or semantic similarity is only used as a fallback
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import Levenshtein

from dialog_acts import BASE_DIR

RESTAURANT_FILE = os.path.join(BASE_DIR, "data", "restaurant_info.csv")
SLOTS = ["area", "food", "pricerange"]
MATCHERS = ["levenshtein", "semantic"]

MAX_EDIT_DISTANCE = 2        # upper bound 
SEMANTIC_THRESHOLD = 0.60    # minimum cosine similarity, tuned with compare_matchers.py --sweep


def _load_slot_values(path=RESTAURANT_FILE):
    """All possible values per slot, taken from the restaurant database."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return {slot: sorted({str(v).strip().lower() for v in df[slot].dropna() if str(v).strip()}) for slot in SLOTS}


SLOT_VALUES = _load_slot_values()

# Words that mean an known value but are not written the same way
SYNONYMS = {
    "center": ("area", "centre"),
    "central": ("area", "centre"),
    "middle": ("area", "centre"),
    "moderately": ("pricerange", "moderate"),
    "asian": ("food", "asian oriental"),
    "oriental": ("food", "asian oriental"),
}

# Every keyword (known value or synonym) with its slot and value, longest first so that
# "north american" is matched before "north"
def _build_keywords():
    return sorted(
        [(value, slot, value) for slot in SLOTS for value in SLOT_VALUES[slot]]
        + [(word, slot, value) for word, (slot, value) in SYNONYMS.items()],
        key=lambda k: -len(k[0].split()))


KEYWORDS = _build_keywords()


def use_restaurant_file(path):
    """Take the known values from another restaurant database (e.g. one given with --restaurants)."""
    global SLOT_VALUES, KEYWORDS, _center
    SLOT_VALUES = _load_slot_values(path)
    KEYWORDS = _build_keywords()
    _value_embeddings.clear()  # The semantic matcher has to embed the new values
    _center = None

# Common English words that are never a preference, so they are skipped to avoid any issues
STOPWORDS = set("""
a an the any some good nice great that this other another different same new particular specific
i im me my you your we it its is are be was what which where how there their they them this that
of for in on at to from with and or but if so just really very quite pretty also too please
restaurant restaurants food place places town part area side price prices priced range kind type sort
want wanted need like looking look find get give serve serves serving would could can will should
dont doesnt not no yes yeah okay ok um uh hmm well fine best cheapest local little something anything
else more one ones about thanks thank bye goodbye hello again sorry
""".split())

# Phrases that mean "I don't care" about one specific slot (area, pricerange, food)
DONTCARE_PATTERNS = {
    "area": r"\bany ?(area|part|where|location|place)\b|\bany part of town\b",
    "pricerange": r"\bany (price|range|cost)\b",
    "food": r"\bany (kind|type|sort)( of)? food\b|\bany (food|cuisine)\b",
}
# Phrases that mean "I don't care" without naming a slot, they apply to the slot the system asked about
DONTCARE_GENERIC = (r"\b(dont care|do not care|doesnt matter|does not matter|dont mind|do not mind|"
                    r"whatever|no preference|anything|either)\b|^(any|uh any|um any)$|\bany (is|will|one)\b")
# The slot a generic "don't care" is about, if the user names it ("i dont care about the price")
SLOT_NOUNS = {"price": "pricerange", "expensive": "pricerange", "area": "area", "part": "area",
              "where": "area", "location": "area", "food": "food", "cuisine": "food"}

# Words that ask for a property of the suggested restaurant
REQUEST_PATTERNS = {
    "phone": r"\b(phone|number|telephone)\b",
    "addr": r"\b(address|where is it|where is that|located)\b",
    "postcode": r"\b(postcode|post code|postal|zip)\b",
    "food": r"\b(what|which) (kind|type|sort)( of)? food\b|\bwhat food\b|\bfood type\b",
    "area": r"\b(what|which) (area|part)\b",
    "pricerange": r"\bprice\b|\bhow (expensive|much)\b|\bcost\b",
}


@dataclass
class Extraction:
    """Result of slot extraction on one utterance."""
    slots: dict = field(default_factory=dict)      # slot -> value that is certain
    uncertain: dict = field(default_factory=dict)  # slot -> (typed word, suggested value), needs confirmation
    unknown: dict = field(default_factory=dict)    # slot -> word that looks like a value but is not in the database
    negated: dict = field(default_factory=dict)    # slot -> value the user does not want ("not chinese")
    requests: list = field(default_factory=list)   # restaurant properties the user asks for

    def found_preference(self):
        return bool(self.slots or self.uncertain or self.unknown or self.negated)


def normalize(utterance):
    """Lower case and split into words, "don't" becomes "dont" like in the DSTC2 data."""
    text = utterance.lower().replace("'", "").replace("’", "")
    return re.findall(r"[a-z0-9]+", text)


# Keyword matching

def _keyword_match(words):
    """Find known values and synonyms, returns ({slot: value}, {slot: value} negated, used word indices)."""
    found, negated, used = {}, {}, set()
    for keyword, slot, value in KEYWORDS:
        kw = keyword.split()
        for i in range(len(words) - len(kw) + 1):
            span = range(i, i + len(kw))
            if words[i:i + len(kw)] != kw or used.intersection(span):
                continue
            used.update(span)
            # "not chinese" or "dont want chinese" is a negation, not a preference
            if "not" in words[max(0, i - 2):i] or "dont" in words[max(0, i - 2):i]:
                negated[slot] = value
            else:
                found[slot] = value
    return found, negated, used


def _pattern_variables(words, used):
    """Words in a slot pattern such as "{x} food" or "in the {x}", returns a list of (word, slot or None)."""
    variables = []
    for i, word in enumerate(words):
        if i in used or word in STOPWORDS or word.isdigit() or len(word) < 3:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else None
        prev = words[i - 1] if i > 0 else None
        if nxt in ("food", "cuisine") or prev in ("serving", "serves", "serve"):
            variables.append((word, "food"))
        elif nxt in ("part", "side", "area") or words[max(0, i - 2):i] == ["in", "the"]:
            variables.append((word, "area"))
        elif nxt in ("priced", "price", "prices"):
            variables.append((word, "pricerange"))
        elif nxt in ("restaurant", "restaurants", "place") or prev == "something":
            variables.append((word, None))  # Could be any slot, e.g. "a fancy restaurant"
    return variables


def _dontcare(text, words, asked_slot):
    """Slots the user doesn't care about."""
    slots = {slot for slot, pattern in DONTCARE_PATTERNS.items() if re.search(pattern, text)}
    if re.search(r"\banywhere\b|\bany where\b", text):
        slots.add("area")
    if not slots and re.search(DONTCARE_GENERIC, text):
        named = {SLOT_NOUNS[w] for w in words if w in SLOT_NOUNS}
        if named:
            slots.update(named)
        elif asked_slot:
            slots.add(asked_slot)
    return slots


def _requests(text):
    return [slot for slot, pattern in REQUEST_PATTERNS.items() if re.search(pattern, text)]


# Fallback matchers

def levenshtein_match(word, slots):
    """Closest known value by edit distance, returns (slot, value, distance) or None if too far away."""
    max_distance = min(MAX_EDIT_DISTANCE, (len(word) + 2) // 3)
    best = None
    for slot in slots:
        for value in SLOT_VALUES[slot]:
            distance = Levenshtein.distance(word, value)
            if distance <= max_distance and (best is None or distance < best[2]):
                best = (slot, value, distance)
    return best


# Frozen DistilBERT gives almost the same embedding to every single word (cosine 0.8-0.95), so:
# - words are embedded in a short context sentence
# - [CLS] and [SEP] are left out of the mean
# - the mean embedding of all known values is subtracted before the cosine similarity
# With these three changes paraphrases like "fancy" -> expensive and "northern" -> north work
EMBEDDING_CONTEXT = "a {} restaurant"
_value_embeddings = {}  # slot -> normalized, centered embeddings of all its values, computed once
_center = None           # mean embedding of all known values


def _raw_embed(words):
    from embeddings import get_embeddings  # Only load DistilBERT when the semantic matcher is used
    return get_embeddings([EMBEDDING_CONTEXT.format(w) for w in words], show_progress=False, skip_special=True)


def _normalize_rows(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def warm_up():
    """Load DistilBERT and embed all known values, so the first user turn isn't slow."""
    global _center
    if _center is not None:
        return
    raw = {slot: _raw_embed(SLOT_VALUES[slot]) for slot in SLOTS}
    _center = np.vstack(list(raw.values())).mean(axis=0)
    for slot in SLOTS:
        _value_embeddings[slot] = _normalize_rows(raw[slot] - _center)


def semantic_match(word, slots):
    """Closest known value by cosine similarity, returns (slot, value, similarity) or None below the threshold."""
    warm_up()
    vector = _normalize_rows(_raw_embed([word]) - _center)[0]
    best = None
    for slot in slots:
        similarities = _value_embeddings[slot] @ vector
        i = int(np.argmax(similarities))
        if similarities[i] >= SEMANTIC_THRESHOLD and (best is None or similarities[i] > best[2]):
            best = (slot, SLOT_VALUES[slot][i], float(similarities[i]))
    return best


def fallback_match(word, slots, matcher):
    if matcher == "semantic":
        return semantic_match(word, slots)
    return levenshtein_match(word, slots)


# Main function

def extract(utterance, asked_slot=None, matcher="levenshtein"):
    """
    Extract preferences and requests from a user utterance.

    asked_slot: the slot the system just asked about, so "any" or "spenish" alone can be understood.
    matcher: fallback when keyword matching fails, "levenshtein" or "semantic".
    """
    if matcher not in MATCHERS:
        raise ValueError(f"matcher must be one of {MATCHERS}")

    words = normalize(utterance)
    text = " ".join(words)
    result = Extraction(requests=_requests(text))

    # 1. Keyword matching
    result.slots, result.negated, used = _keyword_match(words)
    for slot in _dontcare(text, words, asked_slot):
        result.slots.setdefault(slot, "dontcare")

    # 2. Fallback for slots that keyword matching didn't fill
    variables = _pattern_variables(words, used)
    if not variables and asked_slot and not result.found_preference():
        content = [w for w in words if w not in STOPWORDS and not w.isdigit() and len(w) >= 3]
        if len(content) == 1:
            variables = [(content[0], asked_slot)]

    for word, slot in variables:
        candidates = [slot] if slot else [s for s in SLOTS if s not in result.slots]
        candidates = [s for s in candidates if s not in result.slots and s not in result.uncertain]
        if not candidates:
            continue
        match = fallback_match(word, candidates, matcher)
        if match is None:
            if slot == "food":
                result.unknown["food"] = word
            continue
        match_slot, value, score = match
        if matcher == "levenshtein" and score == 0:
            result.slots[match_slot] = value
        else:
            result.uncertain[match_slot] = (word, value)

    return result
