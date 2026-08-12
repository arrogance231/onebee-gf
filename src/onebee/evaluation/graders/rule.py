from __future__ import annotations

import re
import string

_ABSTENTION_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i don't have that information",
    "i do not have that information",
    "you haven't told me",
    "you have not told me",
    "i don't recall",
    "i do not recall",
    "i cannot answer",
    "i can't answer",
    "i have no idea",
    "i'm not aware",
    "i am not aware",
    "i don't remember",
    "i do not remember",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def fuzzy_match(prediction: str, gold: str, threshold: float = 0.8) -> bool:
    pred_tokens = set(normalize_text(prediction).split())
    gold_tokens = set(normalize_text(gold).split())
    if not pred_tokens and not gold_tokens:
        return True
    union = pred_tokens | gold_tokens
    intersection = pred_tokens & gold_tokens
    if not union:
        return False
    return len(intersection) / len(union) >= threshold


def entity_f1(
    prediction_entities: list[str], gold_entities: list[str]
) -> dict[str, float]:
    pred_set = {e.lower().strip() for e in prediction_entities}
    gold_set = {e.lower().strip() for e in gold_entities}
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def detect_abstention(response: str) -> bool:
    lower = response.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)
