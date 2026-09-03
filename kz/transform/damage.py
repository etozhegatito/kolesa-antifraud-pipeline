# -*- coding: utf-8 -*-
"""Implementation for the `kz.transform.damage` module."""

import re

DAMAGE_PATTERNS = [
    "ржавчин",
    "рыжик",
    "гнил",
    "сгнил",
    "на запчасти",
    "по запчастям",
    "на разбор",
    "не на ходу",
    "после дтп",
    "после аварии",
    "аварийн",
    "битая",
    "битый",
    "утопленник",
    "требует ремонта",
    "нужен ремонт",
    "требует вложений",
    "нужны вложения",
    "вложения",
    "под восстановление",
    "не заводится",
    "не работает",
    "не включается",
    "двигатель стучит",
    "мотор стучит",
    "был на ходу",
    "была на ходу",
    "был находу",
    "была находу",
    "есть вложения",
    "с вложениями",
    "кузову вложений",
    "вложений по",
    "без двигателя",
    "нет двигателя",
    "без мотора",
    "без матора",
    "нет мотора",
    "нет матора",
    "без коробки",
    "нет коробки",
    "без кпп",
    "без акпп",
    "без документов",
    "нет документов",
    "снята с учета",
    "снята с учёта",
]


_NEG_BEFORE = {
    "не",
    "нет",
    "нету",
    "без",
    "ни",
    "никаких",
    "никакой",
    "никаким",
    "отсутствует",
    "отсутствуют",
}


_NEG_AFTER = re.compile(
    r"^\s*(не\s+(требует|требуется|нужны|нужно|надо|было|имеет)"
    r"|нет|нету|отсутству\w*)\b"
)

_TOKEN = re.compile(r"[а-яё\d%]+")


def find_damage_keywords(text) -> list[str]:
    """Implement `find_damage_keywords`."""
    t = str(text or "").lower()
    hits = []
    for p in DAMAGE_PATTERNS:
        skip_before = p.split()[0] in _NEG_BEFORE
        for m in re.finditer(re.escape(p), t):
            before = _TOKEN.findall(t[: m.start()])[-2:]
            if not skip_before and any(w in _NEG_BEFORE for w in before):
                continue
            if _NEG_AFTER.match(t[m.end() :]):
                continue
            hits.append(p)
            break
    return sorted(set(hits))


def has_damage(text) -> bool:
    return bool(find_damage_keywords(text))
