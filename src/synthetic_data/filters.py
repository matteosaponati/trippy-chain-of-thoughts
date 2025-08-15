# filters.py
import re, math
import html
import logging

logger = logging.getLogger(__name__)

TRIP_BEFORE_RE = re.compile(r"<trip_before>\s*(.*?)\s*</trip_before>", re.DOTALL | re.IGNORECASE)
ANSWER_RE      = re.compile(r"<answer>\s*(.*?)\s*</answer>",           re.DOTALL | re.IGNORECASE)
END_RE         = re.compile(r"<end>\s*(.*?)\s*(?:</end>)?\s*\Z",        re.DOTALL | re.IGNORECASE)

def _clean(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"^\s*```.*?$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*```$", "", t, flags=re.MULTILINE)
    return t.strip()

def is_parsed(text: str):
    """return (trip_before, answer, end) or (None, None, None) on failure."""
    t = _clean(text)
    if not t:
        return None, None, None

    b = TRIP_BEFORE_RE.search(t)
    a = ANSWER_RE.search(t)
    e = END_RE.search(t)

    if not (b and a and e):
        return None, None, None

    return b.group(1).strip(), a.group(1).strip(), e.group(1).strip()

def normalize_answer(s: str):
    s = s.strip()
    s = s.replace(",", "")
    return s

def is_correct(gold: str, pred: str) -> bool:
    g, p = normalize_answer(gold), normalize_answer(pred)
    try:
        return math.isclose(float(g), float(p), rel_tol=1e-6, abs_tol=1e-6)
    except:
        return g.lower() == p.lower()

def is_length(trip: str, min_c = 30, max_c = 900) -> bool:
    return trip is not None and min_c <= len(trip) <= max_c