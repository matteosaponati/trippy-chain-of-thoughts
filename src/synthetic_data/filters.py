# filters.py
import re, math
import logging

logger = logging.getLogger(__name__)

TRIP_RE = re.compile(r"<trip>\s*(.*?)\s*</trip>", re.DOTALL | re.IGNORECASE)
FINAL_RE = re.compile(r"Final:\s*(.+?)(?:\n|$)", re.MULTILINE | re.IGNORECASE)

def parse_output(text: str):
    """parse the output to extract trip and final answer."""
    if not text:
        return None, None
    
    trip_match = TRIP_RE.search(text)
    final_match = FINAL_RE.search(text)
    
    trip = trip_match.group(1).strip() if trip_match else None
    final = final_match.group(1).strip() if final_match else None
    
    return trip, final

def normalize_answer(s: str):
    s = s.strip()
    s = s.replace(",", "")
    return s

def is_correct(gold: str, pred: str) -> bool:
    g, p = normalize_answer(gold), normalize_answer(pred)
    # numeric?
    try:
        return math.isclose(float(g), float(p), rel_tol=1e-6, abs_tol=1e-6)
    except:
        # text/MCQ exact compare (case-insensitive)
        return g.lower() == p.lower()

def length_ok(trip: str, min_c = 30, max_c = 900) -> bool:
    return trip is not None and min_c <= len(trip) <= max_c