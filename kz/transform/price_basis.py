# -*- coding: utf-8 -*-
"""Classify what the advertised listing price actually represents.

Marketplace descriptions may show several prices for the same vehicle.  The
classifier links the saved listing price to a nearby textual cue instead of
treating any mention of credit or customs clearance as sufficient evidence.
"""

from __future__ import annotations

import re

PRICE_BASIS_VALUES = (
    "cash_customs_cleared",
    "cash_uncleared",
    "credit_price",
    "down_payment",
    "ambiguous",
)
NON_COMPARABLE_PRICE_BASES = frozenset({"cash_uncleared", "credit_price", "down_payment"})

_AMOUNT = re.compile(
    r"(?<![\d.,])(?:"
    r"(?P<grouped>\d{1,3}(?:[\s\u00a0.,]\d{3}){1,2})"
    r"|(?P<number>\d+(?:[.,]\d+)?)"
    r")\s*(?P<unit>млн\.?|миллион\w*|тыс\.?|тысяч\w*)?"
    r"\s*(?:₸|тг\.?|тенге)?(?!\d)",
    re.IGNORECASE,
)

_CUES = {
    "cash_uncleared": re.compile(
        r"(?:\bбез\s+(?:уч[её]та\s+)?рас{1,2}т[ао]мож\w*"
        r"|\bне\s*рас{1,2}т[ао]мож\w*|\bнерас{1,2}т[ао]мож\w*"
        r"|\bрас{1,2}т[ао]мож\w*\s+не\s+(?:включен\w*|оплачен\w*)"
        r"|\bне\s+(?:включен\w*|оплачен\w*)\s+рас{1,2}т[ао]мож\w*)",
        re.IGNORECASE,
    ),
    "cash_customs_cleared": re.compile(
        r"(?:\bс\s+рас{1,2}т[ао]мож\w*"
        r"|\bс\s+уч[её]том\s+(?:доставки\s+и\s+)?рас{1,2}т[ао]мож\w*"
        r"|\b(?:цена\s+)?включа\w*\s+рас{1,2}т[ао]мож\w*"
        r"|\bрас{1,2}т[ао]мож\w*\s+(?:включен\w*|оплачен\w*)"
        r"|\bрас{1,2}т[ао]мож\w*\s*[,;]?\s*ндс\s+(?:включен\w*|оплачен\w*)"
        r"|\bрас{1,2}т[ао]мож\w*\s+утил\w*\s+вс[её]\s+оплачен\w*"
        r"|\bрас{1,2}т[ао]можен(?:а|о|ы)?\b"
        r"|\b(?:цена\s+(?:указана\s+)?)?под\s+ключ\b"
        r")",
        re.IGNORECASE,
    ),
    "credit_price": re.compile(
        r"(?:\b(?:цена\s+)?в\s+кредит\w*|\bкредитн\w*\s+цена)",
        re.IGNORECASE,
    ),
    "down_payment": re.compile(
        r"(?:\bпервоначальн\w*\s+(?:взнос|плат[её]ж)\w*"
        r"|\bперв(?:ый|ого)\s+взнос\w*|\bпв\b)",
        re.IGNORECASE,
    ),
}

_NEGATIVE_CUSTOMS_VALUES = {"нет", "не указан", "не указано", "-"}
_POSITIVE_CUSTOMS_VALUES = {"да", "растаможен", "растаможена"}
_MAX_CUE_DISTANCE = 48


def _parse_amount(match: re.Match) -> float | None:
    grouped = match.group("grouped")
    unit = (match.group("unit") or "").lower()
    if grouped:
        value = float(re.sub(r"\D", "", grouped))
    else:
        raw = (match.group("number") or "").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            return None
    if unit.startswith(("млн", "миллион")):
        value *= 1_000_000
    elif unit.startswith(("тыс", "тысяч")):
        value *= 1_000
    return value if value >= 100_000 else None


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _cue_spans(text: str) -> list[tuple[tuple[int, int], str]]:
    spans = []
    for label, pattern in _CUES.items():
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 12) : match.start()]
            if label == "cash_customs_cleared" and re.search(r"\bне\s*$", prefix):
                continue
            spans.append((match.span(), label))
    return spans


def _labelled_amounts(text: str) -> list[tuple[float, str]]:
    cue_spans = _cue_spans(text)
    out: list[tuple[float, str]] = []
    for amount_match in _AMOUNT.finditer(text):
        amount = _parse_amount(amount_match)
        if amount is None:
            continue
        before = [
            (_distance(amount_match.span(), cue_span), label)
            for cue_span, label in cue_spans
            if cue_span[1] <= amount_match.start()
            and _distance(amount_match.span(), cue_span) <= _MAX_CUE_DISTANCE
        ]
        candidates = before
        if not candidates:
            candidates = [
                (_distance(amount_match.span(), cue_span), label)
                for cue_span, label in cue_spans
                if cue_span[0] >= amount_match.end()
                and label not in {"credit_price", "down_payment"}
                and _distance(amount_match.span(), cue_span) <= _MAX_CUE_DISTANCE
                and not re.search(r"[.;,•·\n]", text[amount_match.end() : cue_span[0]])
            ]
        if not candidates:
            continue
        nearest = min(distance for distance, _ in candidates)
        labels = {label for distance, label in candidates if distance == nearest}
        out.append((amount, labels.pop() if len(labels) == 1 else "ambiguous"))
    return out


def classify_price_basis(
    text: object,
    customs_cleared: object = None,
    listing_price: object = None,
) -> str:
    """Return the strongest supported interpretation of the listing price."""
    source = str(text or "").lower().replace("\u00a0", " ")
    try:
        target = float(listing_price)
    except (TypeError, ValueError):
        target = 0.0

    labelled = _labelled_amounts(source)
    if target > 0:
        tolerance = max(10_000.0, target * 0.01)
        matched = {label for amount, label in labelled if abs(amount - target) <= tolerance}
        if len(matched) == 1:
            return matched.pop()
        if len(matched) > 1:
            return "ambiguous"
    elif len(labelled) == 1:
        return labelled[0][1]

    global_labels = {label for _, label in _cue_spans(source)}
    has_uncleared = "cash_uncleared" in global_labels
    has_cleared = "cash_customs_cleared" in global_labels
    customs = str(customs_cleared or "").strip().lower()
    if customs in _NEGATIVE_CUSTOMS_VALUES:
        return "ambiguous" if has_cleared and not has_uncleared else "cash_uncleared"
    if customs in _POSITIVE_CUSTOMS_VALUES:
        return "ambiguous" if has_uncleared and not has_cleared else "cash_customs_cleared"

    if has_uncleared and not has_cleared:
        return "cash_uncleared"
    if has_cleared and not has_uncleared:
        return "cash_customs_cleared"
    return "ambiguous"


def is_training_eligible(price_basis: object) -> bool:
    """Keep unknown ordinary listings but reject known incomparable targets."""
    return str(price_basis or "") not in NON_COMPARABLE_PRICE_BASES
