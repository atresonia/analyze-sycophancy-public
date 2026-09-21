"""Sycophancy lexicon used to select candidate posts/comments (Route L).

Three tiers, kept as a label rather than a filter:
- tier 1: jargon that names the behaviour ("sycophant", "glazing", "yes-man", ...)
- tier 2: lay phrases that describe the behaviour without naming it
- tier 3: adjacent discourse (AI psychosis, delusion, emotional dependency) - about the harm rather than
  the behaviour; flagged separately so the SAE can be trained with or without it

Each tier has a V1 list (the original hand-written lexicon) and a V2 list (terms added after the log-odds
ranking in build_lexicon.py). match() reports whether a text would have matched the V1 lexicon alone, so the
v1 and v2 corpora can be compared from a single embedding run. Terms are regex fragments, matched
case-insensitively. See docs/lexicon.md for hit counts, precision spot-checks and rejected terms.
"""

import re

# object of the verb, used to keep broad verbs ("validates", "compliments", "affirms") to the chatbot-to-user sense
_OBJ = r"(me|you|us|them|him|her|people|users?|everyone|everything|whatever|anything|my|your|their|the user'?s?)"

TIER1_V1 = [
    r"sycophan",
    r"glaz(e|ing|ed|er)s?\b(?! over)",  # "glazes over the details" is a different sense
    r"yes.?m[ae]n",
    r"ass.?kiss", r"kiss.?ass",
    r"brown.?nos",
    r"boot.?lick",
    r"suck.?up",
    r"flatter",
    r"obsequious",
    r"s[ií].?buey",      # Spanish
    r"bajula",           # Portuguese (bajulação)
]
TIER1_V2 = [
    r"\bfawn",
    r"\bpander",
]

TIER2_V1 = [
    r"tells? (me|you|people|users?) what (i|you|they) want to hear",
    r"agrees? with everything",
    r"just agrees",
    r"blow(ing|s)? smoke",
    # "echo chamber" excluded: ~4k hits, mostly about Reddit/politics rather than the chatbot
    r"hype ?man",
    r"cheerleader",
    r"validat(es|ion) (everything|machine)",
    r"pat on the (back|head)",
    r"strok(e|es|ing) (my|your|the user'?s?) ego",
    r"feed(ing|s)? (my|your) ego",
    r"people.?pleas",
    r"(too|overly|so) agreeable",
    r"sugar.?coat",
    r"dick.?rid",
]
TIER2_V2 = [
    r"agree(s|ing)? with (everything|whatever|anything)",
    r"agreeable",        # also matches agreeableness / disagreeable
    r"push.?back",
    rf"validat(es|ing|ed) {_OBJ}",
    r"(seek|seeks|seeking|need|needs|needed|want|wants|crave|craves|craving|constant|endless|empty|emotional|external|unconditional|blind|instant|cheap) validation",
    r"validation (machine|seeking|bot|loop|engine)",
    rf"(?<!gender[- ])affirm(s|ing|ed)? {_OBJ}",
    r"(constant|endless|empty|unconditional|blind)(ly)? affirm",
    rf"compliment(s|ing|ed)? {_OBJ}",
    r"(constant|endless|empty|excessive|unsolicited|random) compliments",
    r"brutal(ly)? honest",
    r"engage warmly|warmly yet|grounded honesty",  # OpenAI's April-2025 system-prompt wording, quoted in the rollback discussion
]

TIER3_V1 = []
TIER3_V2 = [
    r"psychosis|psychotic",
    r"delusion",
    r"emotional(ly)? dependen",
]

TIERS = {1: TIER1_V1 + TIER1_V2, 2: TIER2_V1 + TIER2_V2, 3: TIER3_V1 + TIER3_V2}
V1_TERMS = set(TIER1_V1 + TIER2_V1 + TIER3_V1)
ANY_RE = re.compile("|".join(t for terms in TIERS.values() for t in terms), re.I)
_TERM_RES = [(tier, t, re.compile(t, re.I)) for tier, terms in TIERS.items() for t in terms]


def match(text: str) -> tuple[int, list[str], bool]:
    """Return (tier, matched terms, in_v1) for a text: the lowest tier among matched terms (1 = jargon,
    2 = phrase only, 3 = adjacent only) or 0 if nothing matched, and whether any V1 term matched."""
    if not ANY_RE.search(text):
        return 0, [], False
    hits = [(tier, term) for tier, term, rx in _TERM_RES if rx.search(text)]
    terms = [term for _, term in hits]
    return min(t for t, _ in hits), terms, any(t in V1_TERMS for t in terms)
