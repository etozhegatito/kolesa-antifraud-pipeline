# -*- coding: utf-8 -*-
"""Советы продавцу по фотографиям объявления.

ПОЧЕМУ ЭТО ПОЯВИЛОСЬ ПОСЛЕ ТРЁХ «ФОТО НЕ ПОМОГАЮТ». Три замера подряд
показали, что признаки из фотографий не улучшают предсказание цены, и на
этом тема была закрыта. Закрыта неправильно: проверялся ровно один вопрос —
даёт ли фото что-то СВЕРХ возраста и цены. Ответ «нет» верен для модели
цены и не имеет отношения к другим применениям.

Сказать продавцу «машина на фотографиях грязная, помойте и переснимите»
полезно независимо от того, коррелирует ли грязь с возрастом. Здесь не
нужно бить возраст — нужно быть правым. А это измерено:

    clip_dirty против бейджа «Аварийная»   AUC 0,948
    clip_rusty против бейджа               AUC 0,935
    clip_rusty против damage-слов, 95%     [0,705; 0,866]

Ржавчина и грязь распознаются надёжно. Повреждения (`clip_damaged`) — нет,
доверительный интервал [0,480; 0,731] накрывает монетку, поэтому советов
про вмятины здесь НЕТ и не будет, пока не появится ручная разметка.

ПОРОГИ БЕРУТСЯ ИЗ СВОИХ ДАННЫХ. «Тёмная фотография» — понятие относительное:
абсолютная яркость зависит от того, как снимают машины вообще, а не от
теории. Поэтому порог — процентиль по корпусу, и совет формулируется
проверяемо: «темнее, чем у четырёх из пяти объявлений». Такое утверждение
можно опровергнуть, в отличие от «фото плоховаты».

Запуск: python -m kz.ml.photo_advice            советы по всем объявлениям
        python -m kz.ml.photo_advice --validate  проверка на просмотрах
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# Доля объявлений, которая считается «хуже обычного». Двадцать процентов, а
# не десять: совет должен доставаться заметной части людей, иначе инструмент
# молчит и кажется сломанным. И не половине — иначе он теряет смысл.
WORSE_THAN = 0.20

# Минимум кадров, чтобы судить об объявлении. По одной фотографии вывод о
# «тёмных снимках» был бы гаданием.
MIN_PHOTOS = 2


def load_photo_signals() -> pd.DataFrame:
    """Оценки CLIP и метрики качества, свёрнутые до одного объявления.

    Берётся среднее по кадрам, а не максимум: речь о впечатлении от
    объявления в целом. Для повреждений максимум был бы правильнее (хватает
    одного кадра с помятым крылом), но советов про повреждения мы не даём.
    """
    from kz.ml.photo_clip import load as load_clip

    clip = load_clip()
    agg = clip.groupby("ad_id").agg(
        clip_dirty=("clip_dirty", "mean"),
        clip_rusty=("clip_rusty", "mean"),
        clip_studio=("clip_studio", "mean"),
        n_photos=("clip_dirty", "size"),
    ).reset_index()

    try:
        from kz.ml.photo_features import load_quality
        agg = agg.merge(load_quality(), on="ad_id", how="left")
    except Exception:                       # noqa: BLE001 — метрик может не быть
        pass
    return agg


def thresholds(df: pd.DataFrame, cols: list[str],
               worse_than: float = WORSE_THAN) -> dict:
    """Границы «хуже, чем у большинства» — процентили по корпусу.

    Для яркости и резкости плохо быть НИЗКО, поэтому берётся нижний
    процентиль; для грязи и ржавчины плохо быть ВЫСОКО — верхний.
    """
    low_is_bad = {"img_brightness", "img_sharpness", "img_contrast"}
    out = {}
    for c in cols:
        if c not in df:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) < 50:
            continue
        q = worse_than if c in low_is_bad else 1 - worse_than
        out[c] = float(np.quantile(v, q))
    return out


def advise(row: pd.Series, cuts: dict) -> list[str]:
    """Советы по одному объявлению. Пустой список — значит всё в порядке."""
    out = []
    pct = int(WORSE_THAN * 100)

    if row.get("n_photos", 0) < MIN_PHOTOS:
        return ["Слишком мало фотографий, чтобы судить о съёмке."]

    if "img_brightness" in cuts and row.get("img_brightness", np.inf) < cuts["img_brightness"]:
        out.append(f"Фотографии темнее, чем у {100-pct}% объявлений. "
                   f"Переснимите днём или в освещённом месте.")
    if "img_sharpness" in cuts and row.get("img_sharpness", np.inf) < cuts["img_sharpness"]:
        out.append(f"Снимки менее резкие, чем у {100-pct}% объявлений. "
                   f"Протрите объектив и снимайте с упора.")
    if "clip_dirty" in cuts and row.get("clip_dirty", -np.inf) > cuts["clip_dirty"]:
        out.append("Машина на фотографиях выглядит грязной. Мойка перед "
                   "съёмкой — самое дешёвое, что можно сделать для объявления.")
    if "clip_rusty" in cuts and row.get("clip_rusty", -np.inf) > cuts["clip_rusty"]:
        out.append("На снимках заметны следы коррозии. Если они есть — "
                   "напишите об этом в описании: необъяснённая дешевизна "
                   "читается как приманка, а честно названный дефект нет.")
    return out


def validate(log=print) -> None:
    """Проверка на просмотрах: связаны ли «плохие фото» с интересом к объявлению.

    ИТОГ ПРОВЕРКИ: она НЕ УДАЛАСЬ, и это записано здесь, а не спрятано.

    Ожидание было такое: если совет осмыслен, объявления с «плохими» фото
    должны собирать меньше просмотров. Получилось наоборот — внутри ценовых
    групп они собирают на 30-44% БОЛЬШЕ.

    Первый подозреваемый — накопление просмотров со временем — не
    подтвердился. Нормировка на срок ничего не изменила, потому что срок
    здесь почти всегда единица: `days_up` меряет не «сколько висит на
    рынке», а «сколько прошло от публикации до нашей первой встречи», а
    ловим мы объявления почти сразу.

    Остаются необъяснённые кандидаты: внутри ценовой группы дешёвые машины
    смотрят чаще, а плохие фотографии чаще у дешёвых; популярные массовые
    модели одновременно ищут больше и снимают хуже.

    ЧТО ИЗ ЭТОГО СЛЕДУЕТ ДЛЯ ПРОДУКТА. Советы остаются, но обещание
    меняется. Утверждение «ваши фотографии темнее, чем у 80% объявлений» —
    ФАКТ по построению, его можно проверить. Утверждение «переснимите, и
    вас будут смотреть чаще» — НЕ доказано нашими данными и в интерфейсе не
    произносится.

    Доказать его можно было бы только экспериментом: сменить фотографии у
    случайной половины объявлений и сравнить. Наблюдением — нельзя.
    """
    from kz.core.db import get_engine

    from kz.ml.survival import parse_posted

    sig = load_photo_signals()
    cd = pd.read_sql(
        "SELECT ad_id, views_count, price_tenge, age, photos_count, "
        "posted_date, scraped_at FROM clean_data", get_engine(),
        dtype={"ad_id": str})
    d = sig.merge(cd, on="ad_id", how="inner")
    d["views_count"] = pd.to_numeric(d.views_count, errors="coerce")

    # Просмотры НАКАПЛИВАЮТСЯ: объявление, висящее месяц, наберёт больше
    # вчерашнего просто по времени. Первый замер этого не учитывал и дал
    # обратный результат — «плохие фото собирают больше просмотров», — что
    # было измерением срока размещения, а не качества съёмки.
    start = pd.to_datetime(d.posted_date.map(parse_posted))
    seen = pd.to_datetime(d.scraped_at, errors="coerce")
    d["days_up"] = (seen - start).dt.days.clip(lower=1)
    d["views_per_day"] = d.views_count / d.days_up

    d = d[d.views_per_day.notna() & d.price_tenge.notna() & d.days_up.notna()]
    if len(d) < 200:
        log(f"Объявлений с фото и просмотрами: {len(d)} — мало для проверки")
        return

    cols = ["img_brightness", "img_sharpness", "clip_dirty", "clip_rusty"]
    cuts = thresholds(d, cols)
    d["advice_n"] = d.apply(lambda r: len(advise(r, cuts)), axis=1)

    log(f"Объявлений с фотографиями и просмотрами: {len(d)}")
    log(f"Пороги (процентиль {int(WORSE_THAN*100)}% по корпусу):")
    for c, v in cuts.items():
        log(f"   {c:16} {v:.3f}")

    log(f"\nМедианный срок размещения: {d.days_up.median():.0f} дн. "
        f"(от {d.days_up.min():.0f} до {d.days_up.max():.0f})")

    band = pd.cut(d.price_tenge, [0, 5e6, 10e6, 20e6, np.inf],
                  labels=["<5M", "5-10M", "10-20M", "20M+"])
    for metric, title in [("views_count", "ВСЕГО просмотров (не нормировано)"),
                          ("views_per_day", "просмотров В ДЕНЬ")]:
        log(f"\nМедиана: {title}")
        log(f"   {'группа':8} {'без советов':>12} {'есть советы':>12} {'разница':>9}")
        for name in band.cat.categories:
            m = np.asarray(band == name)
            good = d.loc[m & (d.advice_n == 0), metric]
            bad = d.loc[m & (d.advice_n > 0), metric]
            if len(good) < 20 or len(bad) < 20:
                continue
            g, b = float(good.median()), float(bad.median())
            log(f"   {name:8} {g:12.1f} {b:12.1f} {(b-g)/g*100:8.1f}%")

    # Срок размещения сам по себе: если он различается между группами,
    # первая таблица измеряла его, а не фотографии.
    log("\nМедианный срок размещения по группам совета:")
    for label, sub in [("без советов", d[d.advice_n == 0]),
                       ("есть советы", d[d.advice_n > 0])]:
        log(f"   {label:14} {sub.days_up.median():5.0f} дн.  (n={len(sub)})")

    log("\nЭто наблюдение, а не эксперимент: доказать, что дело именно в "
        "фотографиях,\nможно было бы только сменив их у случайной половины "
        "объявлений.")


def main():
    if "--validate" in sys.argv:
        validate()
        return

    sig = load_photo_signals()
    cuts = thresholds(sig, ["img_brightness", "img_sharpness",
                            "clip_dirty", "clip_rusty"])
    sig["advice"] = sig.apply(lambda r: advise(r, cuts), axis=1)
    n = int((sig.advice.map(len) > 0).sum())
    print(f"Объявлений с фотографиями: {len(sig)}")
    print(f"Есть что посоветовать:     {n} ({n/len(sig)*100:.0f}%)\n")

    counts = pd.Series([a.split(".")[0] for lst in sig.advice for a in lst]
                       ).value_counts()
    print("Какие советы выдаются чаще:")
    for text, k in counts.items():
        print(f"  {k:5}  {text}")

    print("\nПример объявлений с советами:")
    for _, r in sig[sig.advice.map(len) > 0].head(3).iterrows():
        print(f"\n  {r.ad_id} ({int(r.n_photos)} фото):")
        for a in r.advice:
            print(f"    • {a}")


if __name__ == "__main__":
    main()
