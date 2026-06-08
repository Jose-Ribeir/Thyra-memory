"""Cue normalization: NLTK PorterStemmer with irregular-plural override."""

from __future__ import annotations

import threading

_IRREGULARS: dict[str, str] = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "oxen": "ox",
    "leaves": "leaf",
    "loaves": "loaf",
    "halves": "half",
    "knives": "knife",
    "wives": "wife",
    "lives": "life",
    "shelves": "shelf",
    "wolves": "wolf",
    "selves": "self",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "data": "datum",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "analyses": "analysis",
    "bases": "basis",
    "theses": "thesis",
    "diagnoses": "diagnosis",
    "axes": "axis",
    "crises": "crisis",
    "fungi": "fungus",
    "alumni": "alumnus",
    "cacti": "cactus",
}

_stemmer = None
_stemmer_lock = threading.Lock()


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        with _stemmer_lock:
            if _stemmer is None:
                try:
                    from nltk.stem import PorterStemmer

                    _stemmer = PorterStemmer()
                except Exception:
                    _stemmer = False  # sentinel: unavailable
    return _stemmer if _stemmer is not False else None


def normalize_cue(token: str) -> str:
    """Normalize a cue token: irregular map → NLTK stem → fallback simple rules."""
    t = token.lower()
    if t in _IRREGULARS:
        return _IRREGULARS[t]

    stemmer = _get_stemmer()
    if stemmer is not None:
        try:
            return stemmer.stem(t)
        except Exception:
            pass

    # Fallback simple rules (only the safe ones: plurals)
    if t.endswith("ies") and len(t) > 5:
        return t[:-3] + "y"
    if t.endswith("ses") and len(t) > 5:
        return t[:-2]
    if t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
        return t[:-1]
    return t
