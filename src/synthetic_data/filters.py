# filters.py
import re, math

TRIP_RE = re.compile(r"<trip>(.*?)</trip>", re.S | re.I)
FINAL_RE = re.compile(r"Final:\s*(.+)$", re.M | re.I)

def parse_output(text: str):
    trip = TRIP_RE.search(text)
    final = FINAL_RE.search(text)
    return (trip.group(1).strip() if trip else None,
            final.group(1).strip() if final else None)

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

def length_ok(trip: str, min_c = 60, max_c = 900) -> bool:
    return trip is not None and min_c <= len(trip) <= max_c