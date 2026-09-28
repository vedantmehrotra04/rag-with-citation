"""Input and output guardrails.

Two layers, two different jobs:

  injection_match  — a narrow heuristic on user input, layer 1 of defence.
  find_pii         — Presidio, on anything about to reach the user or the 
model.
                                                                          
Neither is the layer we rely on. The approval gate in the graph is an integer
comparison in code, so no text from any source can bypass it. These reduce
much reaches that gate; they do not guarantee anything on their own.
"""                                                                       

import re                                                                 

from presidio_analyzer import AnalyzerEngine                              

# ----------------------------------------------------------------

# Patterns that mean "override your instructions", not "ask for something"
#
# Deliberately NARROW. "skip the approval step" and "without approval" are
# here: a user asking to bypass approval is making a request, and the correct
# answer is to proceed normally and let the gate deny it. Treating a reque
# an attack breaks legitimate users — and a guardrail with false positives is a
# guardrail that gets switched off.                                       
INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",   
    r"disregard\s+(your|all|the|any)\s+(instructions|rules|guidelines|prompt)",
    r"forget\s+(your|all|the)\s+(instructions|rules)",                    
    r"you\s+are\s+now\s+a",
    r"new\s+instructions\s*:",                                            
    r"^\s*system\s*:",
    r"pretend\s+(you\s+are|to\s+be)",                                     

r"(reveal|show|print|repeat)\s+(your|the)\s+(system\s+)?(prompt|instruction)"
)
                                                                          
REFUSAL = (
    "I can't act on that request. If you need a billing credit, tell me th"
    "invoice id and what went wrong and I'll look into it."
)                                                                         

                                                                          
def injection_match(text: str) -> str | None:
    """The pattern that matched, or None.                                 

    Returns the pattern rather than a bool so a failure is debuggable — you see
    which rule fired, not just that one did.
    """                                                                   
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return None


# ---------------------------------------------------------------------- PII

# What counts as PII here. PERSON is the one a regex cannot do — "Tom Byrne"
# is undetectable by pattern and is exactly what redact() is stopping.
#
# LOCATION and DATE_TIME are deliberately excluded: invoice dates and country
# names are not leaks, and flagging them would train people to ignore the check
DEFAULT_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD"
)

# Presidio scores 0-1. Pattern-based entities (email, card) score ~1.0 and card
# numbers are checksum-validated. PERSON comes from NER and sits around 0.85.
# Below ~0.5 you start flagging ordinary capitalised words as names.
SCORE_THRESHOLD = 0.5

_ANALYZER = None


def _analyzer() -> AnalyzerEngine:
    """Built once — constructing it loads a spaCy model and takes a second or
two."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    return _ANALYZER


def find_pii(
    text: str,
    entities: tuple[str, ...] = DEFAULT_ENTITIES,
    threshold: float = SCORE_THRESHOLD,
) -> list[str]:
    """Return what was found, as "TYPE:value" strings. Empty list means clean.

    Returning the values rather than a bool means a failing test tells you what
    leaked, which is the difference between a useful alert and a red light.
    """
    if not text:
        return []

    results = _analyzer().analyze(text=text, entities=list(entities), language="en")

    return [
        f"{r.entity_type}:{text[r.start:r.end]}"
        for r in results
        if r.score >= threshold
    ]