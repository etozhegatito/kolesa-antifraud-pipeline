# -*- coding: utf-8 -*-
"""Насколько свежи данные и чему из отчётов можно верить прямо сейчас.

ЗАЧЕМ ЭТО ОТДЕЛЬНЫМ МОДУЛЕМ. Конвейер писался так, будто его запускают
каждый день. В действительности он живёт на ноутбуке, который закрывают, и
между прогонами проходит неделя-две. За 39 календарных дней сбор шёл 8 —
то есть 79% времени данные просто стояли.

Молча это не проходит. Статус объявления, проверенный две недели назад,
сегодня не факт. Объявление, не встреченное в последнем обходе, могло уйти
с рынка — или просто выпасть за глубину обхода, потому что его вытеснили
новые. Отчёт, который печатает «медианный срок жизни 24 дня», не сообщая,
что статусы устарели на две недели, выдаёт догадку за измерение.

Здесь считаются только факты о возрасте данных. Что с ними делать —
решает каждый отчёт сам: где-то достаточно приписки, где-то честнее
отказаться от вывода.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE = ZoneInfo("Asia/Almaty")


def local_date_from_utc_iso(value: str) -> date:
    """Календарный день артефакта в часовом поясе рынка.

    Метаданные правильно пишутся в UTC. Но ``.date()`` до timezone-конвертации
    превращало обучение в 01:15 Алматы в «вчера», потому что в UTC ещё было
    20:15 предыдущего дня. Старые naive-метки считаем UTC для совместимости.
    """
    created = datetime.fromisoformat(value)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(LOCAL_TIMEZONE).date()


@dataclass
class Freshness:
    """Возраст данных на момент вопроса."""
    last_collect: date | None      # последний день, когда шёл сбор листинга
    collect_days: int              # сколько дней всего собирали
    span_days: int                 # сколько календарных дней прошло
    last_status_check: date | None
    ads_total: int
    ads_status_checked: int
    model_created: date | None

    @property
    def data_age_days(self) -> int | None:
        if self.last_collect is None:
            return None
        return (date.today() - self.last_collect).days

    @property
    def status_age_days(self) -> int | None:
        if self.last_status_check is None:
            return None
        return (date.today() - self.last_status_check).days

    @property
    def model_age_days(self) -> int | None:
        if self.model_created is None:
            return None
        return (date.today() - self.model_created).days

    @property
    def status_coverage(self) -> float:
        """Доля объявлений, чей статус хоть раз проверялся."""
        return self.ads_status_checked / self.ads_total if self.ads_total else 0.0

    @property
    def collect_regularity(self) -> float:
        """Доля дней, в которые сбор реально шёл. 1.0 — каждый день."""
        return self.collect_days / self.span_days if self.span_days else 0.0


def measure() -> Freshness:
    """Один запрос к базе на каждый факт, без вычислений поверх."""
    import pandas as pd

    from kz.core.db import get_engine

    eng = get_engine()

    def scalar(sql: str):
        return pd.read_sql(sql, eng).iloc[0, 0]

    days = pd.read_sql("SELECT DISTINCT seen_date FROM sightings", eng)
    seen = pd.to_datetime(days["seen_date"]).dt.date.tolist() if len(days) else []

    last_check = scalar("SELECT MAX(checked_at) FROM ad_status")
    model_created = None
    try:
        from kz.ml.train_price_model import load_artifact
        _, meta = load_artifact()
        model_created = local_date_from_utc_iso(meta["created_at_utc"])
    except Exception:                       # noqa: BLE001 — артефакта может не быть
        pass

    return Freshness(
        last_collect=max(seen) if seen else None,
        collect_days=len(seen),
        span_days=(max(seen) - min(seen)).days + 1 if seen else 0,
        last_status_check=pd.to_datetime(last_check).date() if last_check else None,
        ads_total=int(scalar("SELECT COUNT(*) FROM clean_data")),
        ads_status_checked=int(scalar("SELECT COUNT(*) FROM ad_status")),
        model_created=model_created,
    )


def report(f: Freshness, log=print) -> None:
    """Шапка «чему верить», которую печатают отчёты перед своими числами."""
    def age(n, what):
        if n is None:
            return f"{what}: никогда"
        if n == 0:
            return f"{what}: сегодня"
        return f"{what}: {n} дн. назад"

    log("Свежесть данных:")
    log(f"  {age(f.data_age_days, 'последний сбор')}, "
        f"{age(f.status_age_days, 'проверка статусов')}, "
        f"{age(f.model_age_days, 'обучение модели')}")
    log(f"  статус проверялся у {f.ads_status_checked} из {f.ads_total} "
        f"объявлений ({f.status_coverage*100:.0f}%)")
    log(f"  сбор шёл {f.collect_days} дней из {f.span_days} "
        f"({f.collect_regularity*100:.0f}% времени)")


def stale_warnings(f: Freshness, status_days: int = 7,
                   coverage: float = 0.5) -> list[str]:
    """Что именно перестаёт быть надёжным при таком возрасте данных.

    Пороги эмпирические и намеренно мягкие: смысл не в том, чтобы запретить
    отчёт, а в том, чтобы рядом с числом стояла причина ему не доверять.
    """
    out = []
    if f.status_age_days is not None and f.status_age_days > status_days:
        out.append(
            f"Статусы проверялись {f.status_age_days} дн. назад. Объявление, "
            f"ушедшее с рынка за это время, всё ещё числится активным — "
            f"сроки жизни завышены, доля ушедших занижена.")
    if f.status_coverage < coverage:
        out.append(
            f"Статус проверялся лишь у {f.status_coverage*100:.0f}% объявлений. "
            f"Остальные записаны активными по умолчанию, а не по проверке.")
    if f.collect_regularity < 0.5 and f.span_days > 7:
        out.append(
            f"Сбор шёл {f.collect_regularity*100:.0f}% дней. В истории цен и "
            f"встреч есть провалы, и события внутри них не датируются точно.")
    return out
