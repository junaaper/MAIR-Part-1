from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: int
    antecedent: tuple
    consequent: str
    value: bool
    reason: str


RULES = [
    Rule(1, (("pricerange", "cheap"), ("food quality", "good")), "touristic", True,
         "it is cheap and has good food"),
    Rule(2, (("food", "romanian"),), "touristic", False,
         "romanian food is unfamiliar to most tourists"),
    Rule(3, (("crowdedness", "busy"),), "assigned seats", True,
         "it is busy, so the waiter decides where you sit"),
    Rule(4, (("length of stay", "long"),), "children", False,
         "you stay a long time, which is not advised with children"),
    Rule(5, (("crowdedness", "busy"),), "romantic", False,
         "it is busy"),
    Rule(6, (("length of stay", "long"),), "romantic", True,
         "it allows you to stay for a long time"),
]

CONSEQUENTS = ["touristic", "assigned seats", "children", "romantic"]

PROPERTY_PHRASES = {
    "touristic": ("it is touristic", "it is not touristic"),
    "assigned seats": ("the waiter assigns you a seat", "you can choose your own seat"),
    "children": ("it is suitable for children", "it is not suitable for children"),
    "romantic": ("it is romantic", "it is not romantic"),
}

# When two rules disagree, the positive conclusion wins, so a restaurant isn't ruled out by one conflicting rule
NEGATIVE_WINS = False

# What happens when no rule says anything explicitly
DEFAULTS = {"touristic": False, "assigned seats": False, "children": False, "romantic": False}


def _normalize(value):
    return str(value).strip().lower()


def _matches(cell, expected):
    """True if the database value means the value the rule asks for ("long stay" counts as "long")."""
    cell, expected = _normalize(cell), _normalize(expected)
    return cell == expected or cell.startswith(expected + " ")


def fires(rule, restaurant):
    """True if all conditions of the rule hold for the restaurant."""
    return all(_matches(restaurant.get(column, ""), value) for column, value in rule.antecedent)


def infer(restaurant, rules=RULES):
    """
    Apply all rules that fire on one restaurant (a dict, one row of restaurant_info_extended.csv).
    Returns {consequent: [(value, rule), ...]} with every conclusion and the rule behind it.
    A consequent with no fired rule is left out; one with both True and False is a contradiction,
    which is resolved later (this function only collects the conclusions).
    """
    derived = {}
    for rule in rules:
        if fires(rule, restaurant):
            derived.setdefault(rule.consequent, []).append((rule.value, rule))
    return derived


def contradictions(derived):
    """The consequents that were derived as both True and False."""
    return [prop for prop, conclusions in derived.items() if len({value for value, _ in conclusions}) > 1]


@dataclass
class Conclusion:
    """What the system concludes about one property of one restaurant."""
    prop: str
    value: bool
    supporting: tuple
    conflicting: tuple
    derived: bool

    @property
    def contradicted(self):
        return bool(self.conflicting)


def resolve(restaurant, prop, rules=RULES):
    """
    The value of one property for one restaurant, with contradictions resolved.
    When rules disagree the strategy in NEGATIVE_WINS decides, and the rules on the losing side
    are kept in the Conclusion so the system can tell the user about the conflict.
    """
    conclusions = infer(restaurant, rules).get(prop, [])
    if not conclusions:
        return Conclusion(prop, DEFAULTS.get(prop, False), (), (), derived=False)
    values = {value for value, _ in conclusions}
    if len(values) > 1:
        winner = not NEGATIVE_WINS
    else:
        winner = next(iter(values))
    supporting = tuple(rule for value, rule in conclusions if value == winner)
    conflicting = tuple(rule for value, rule in conclusions if value != winner)
    return Conclusion(prop, winner, supporting, conflicting, derived=True)


def satisfies(restaurant, requirements, rules=RULES):
    """True if the restaurant has every property the user asked for, with the value they asked for."""
    return all(resolve(restaurant, prop, rules).value == value for prop, value in requirements.items())


def explain(restaurant, requirements, rules=RULES, mention_conflict=True):
    """
    Why the restaurant meets the user's additional requirements, as one sentence part per property,
    e.g. "it is romantic because it allows you to stay for a long time".
    """
    parts = []
    for prop, value in requirements.items():
        conclusion = resolve(restaurant, prop, rules)
        phrase = PROPERTY_PHRASES[prop][0 if conclusion.value else 1]
        if conclusion.supporting:
            reasons = " and ".join(rule.reason for rule in conclusion.supporting)
            part = f"{phrase} because {reasons}"
        else:
            part = f"{phrase}, since none of the rules say otherwise"
        if mention_conflict and conclusion.conflicting:
            losing = " and ".join(rule.reason for rule in conclusion.conflicting)
            part += f" (note: one rule suggested otherwise, because {losing})"
        parts.append(part)
    return ", and ".join(parts)