from typing import List, Dict, Optional
from collections import Counter
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
    t = re.sub(r"^\s*```.*?$", "", t, flags = re.MULTILINE)
    t = re.sub(r"^\s*```$", "", t, flags = re.MULTILINE)
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

def self_consistency(outputs: List[str], 
                    gold_answer: str) -> Optional[tuple]:
        """Select the best output based on self-consistency."""
        if not outputs or not gold_answer:
            return None

        best = None
        best_score = float("-inf")

        for i, text in enumerate(outputs or []):
            
            trip_before, answer, end = is_parsed(text or "")
            if not (trip_before and answer and end):
                continue
            if not is_length(trip_before): 
                continue
            if not is_length(end): 
                continue
            if not is_correct(gold_answer, answer): 
                continue
            
            score = len(trip_before) + 0.5 * len(end)
            if score > best_score:
                best_score = score
                best = (trip_before, answer, end, text)
    
        return best

def evaluate_answer(text: str):
    """Extract numerical answer from model output"""

    patterns = [r"[Tt]he answer is[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"[Aa]nswer[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s*$|\s*\.$)"]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].replace(",", "")
            try:
                return answer
            except:
                continue
    return None

def majority_vote(outputs: List[str]):
        
        answers = []
        for output in outputs:
            answer = evaluate_answer(output)
            if answer is not None:
                answers.append(answer)
        if not answers:
            return None
        
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)
        return most_common[0][0] if most_common else None