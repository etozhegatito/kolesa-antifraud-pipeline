"""
Job 2: чистка и разметка подозрительных объявлений (clean.py).

Философия (проговаривай это на собеседовании — это и есть «зрелость»):
  1. RAW не трогаем. Читаем raw_data.csv → пишем clean_data.csv.
     Ошиблись в правилах → поправили код → перезапустили. Данные целы.
  2. НЕ УДАЛЯЕМ — ПОМЕЧАЕМ. Каждая строка получает is_suspicious и
     список причин. Удаление необратимо и прячет информацию; флаг
     обратим, и сам факт «подозрительности» — признак для моделей.
  3. Пропуск — это тоже данные. Пропуски у нас неслучайные (MNAR):
     продавцы старых машин «забывают» пробег не случайно. Кодируем
     факт пропуска отдельным бинарным признаком.

Запуск: python clean.py
Вход:  raw_data.csv (+ sightings.csv и ad_status.csv, если есть)
Выход: clean_data.csv + отчёт о качестве в консоль
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "clean.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine

LABELS_CSV   = "data/manual_labels.csv"  # правится человеком руками — остаётся файлом
OUT_CSV      = "data/clean/clean_data.csv"  # держим и CSV: дёшево, пригодится для отладки/сверки

# Лексикон «убитости» и поиск с учётом отрицаний — в damage.py
# (единственный источник; раньше список дублировался здесь и в enrich.py).
from damage import DAMAGE_PATTERNS, has_damage as _has_damage

CURRENT_YEAR = date.today().year

# ─── Пороги правил (вынесены наверх: пороги = гиперпараметры чистки,
#     их будем крутить, глядя на precision ручной проверки) ──────────────────
MIN_YEAR            = 1950
MAX_PRICE           = 300_000_000   # дороже — почти наверняка опечатка
MIN_PRICE           = 200_000       # дешевле — металлолом или приманка
MAX_MILEAGE         = 1_000_000
MAX_KM_PER_YEAR     = 100_000       # больше — такси-ад или скрутка «вверх»?
ROBUST_Z_THRESHOLD  = 3.5           # порог модифицированного z-score
MIN_GROUP_SIZE      = 8             # меньше — статистика по группе не значима

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─── Шаг 1. Типизация и нормализация ─────────────────────────────────────────
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим типы и чиним мелочи. Принцип: одна каноническая форма
    для каждого поля — иначе группировки развалятся («Алматы» и
    «Алматы » — это уже две разные группы)."""
    df = df.copy()

    df["price_tenge"] = pd.to_numeric(df["price_tenge"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df["engine_volume"] = pd.to_numeric(df["engine_volume"], errors="coerce")

    for col in ["brand", "model", "city", "engine_type",
                "transmission", "body_type", "condition"]:
        df[col] = df[col].astype("string").str.strip()

    # condition можно частично восстановить из меток сайта:
    # бейдж «Новая» — это тот же факт, из другого места страницы
    is_new_label = df["labels"].fillna("").str.contains("Новая")
    df.loc[df["condition"].isna() & is_new_label, "condition"] = "новый"

    # Возраст: понятнее и полезнее для модели, чем «год выпуска».
    # +1 чтобы у машин текущего года возраст был 1, а не 0
    # (возраст 0 ломает деление пробег/возраст).
    df["age"] = CURRENT_YEAR - df["year"] + 1

    return df


# ─── Шаг 2. Индикаторы пропусков ─────────────────────────────────────────────
def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Пропуски у нас MNAR (missing not at random): вероятность пропуска
    зависит от самого скрытого значения — большой пробег «забывают» чаще.
    Поэтому факт пропуска кодируем отдельным признаком: для модели это
    информация, для антифрода — сигнал."""
    df = df.copy()
    df["is_mileage_missing"] = df["mileage_km"].isna().astype(int)
    df["is_description_missing"] = df["description"].isna().astype(int)
    return df


# ─── Шаг 3. Жёсткие правила валидности (rule-based) ─────────────────────────
def apply_hard_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Правила = проверки на ФИЗИЧЕСКУЮ возможность значений.
    Они не про статистику («редко») — они про невозможность («не бывает»).
    Каждое правило добавляет строку-причину: одна колонка-флаг без причин
    бесполезна при разборе — не поймёшь, ЗА ЧТО помечено."""
    df = df.copy()
    reasons = [[] for _ in range(len(df))]

    def flag(mask: pd.Series, reason: str):
        for i in np.where(mask.fillna(False).to_numpy())[0]:
            reasons[i].append(reason)

    flag(df["year"] < MIN_YEAR, "year_too_old")
    flag(df["year"] > CURRENT_YEAR + 1, "year_in_future")
    flag(df["price_tenge"] < MIN_PRICE, "price_too_low")
    flag(df["price_tenge"] > MAX_PRICE, "price_too_high")
    flag(df["mileage_km"] > MAX_MILEAGE, "mileage_extreme")

    # «б/у, пробег 0» — либо ошибка ввода, либо сокрытие пробега
    flag((df["condition"] == "б/у") & (df["mileage_km"] == 0),
         "used_but_zero_mileage")

    # Пробег/возраст: >100k км/год стабильно много лет — нереалистично
    km_per_year = df["mileage_km"] / df["age"]
    flag(km_per_year > MAX_KM_PER_YEAR, "km_per_year_extreme")

    # Свежая машина с подозрительно малой ценой — классика приманки
    flag((df["age"] <= 4) & (df["price_tenge"] < 4_000_000)
         & (df["condition"] != "новый"), "young_car_cheap")

    df["rule_reasons"] = ["|".join(r) for r in reasons]
    return df


# ─── Шаг 4. Статистические выбросы по цене ───────────────────────────────────
def add_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Идея: выброс — не «дорого/дёшево вообще», а «странно ДЛЯ СВОЕЙ ГРУППЫ».
    G-Class за 80 млн — норма; Camry 2020 за 1 млн — аномалия.

    Математика по шагам:

    1) log_price = ln(price).
       Цены распределены логнормально (длинный правый хвост: медиана 7.6 млн,
       среднее 12+ млн). После логарифма распределение ~симметричное, и
       «одинаковое отклонение» значит «одинаковое отклонение В ПРОЦЕНТАХ»:
       ln(2x) - ln(x) = ln 2 при любом x. Разница в 1 млн для Матиза и
       для G-Class — разные события; разница в 2 раза — одно и то же.

    2) Группа = brand × age_bucket (корзина возраста).
       Почему корзина, а не точный год: по паре (brand='Rover', year=1997)
       у нас 1-2 машины — статистика на двух точках это гадание.
       Корзины [0-3], [4-7], [8-12], [13-20], [21+] дают группам объём.

    3) Модифицированный z-score (робастный аналог обычного):
           z_i = 0.6745 * (x_i - median) / MAD,
       где MAD = median(|x_i - median(x)|) — медианное абсолютное отклонение.

       Расшифровка символов:
         x_i     — log-цена конкретного объявления
         median  — медиана log-цен ГРУППЫ (середина: 50% выше, 50% ниже)
         MAD     — «типичное» отклонение от медианы, тоже через медиану
         0.6745  — константа согласования: для нормального распределения
                   MAD ≈ 0.6745·σ, поэтому умножение приводит шкалу к
                   привычным «сигмам» и порог 3.5 сравним с |z|>3.5σ.

       Почему не классический z=(x-μ)/σ: μ (среднее) и σ (ст. отклонение)
       сами ломаются от выбросов — один Rolls-Royce в группе утащит среднее
       и раздует σ так, что настоящие аномалии станут «нормой». Это называется
       masking (маскировка выбросов). Медиана и MAD устойчивы: до 50%
       мусора в данных им безразличны (breakdown point = 50%, у среднего = 0%).

    4) Порог |z| > 3.5 — рекомендация Иглевича-Хоглина, стандарт для
       модифицированного z-score. Знак важен: z < -3.5 — подозрительно
       ДЁШЕВО (приманка мошенника), z > +3.5 — дорого (скорее опечатка
       или неадекватный продавец, для фрода менее интересно).
    """
    df = df.copy()
    df["log_price"] = np.log(df["price_tenge"])

    bins = [0, 3, 7, 12, 20, np.inf]
    labels = ["0-3", "4-7", "8-12", "13-20", "21+"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels)

    def robust_z(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(0.0, index=s.index)
        return 0.6745 * (s - med) / mad

    # ИЕРАРХИЧЕСКАЯ группировка (fallback-цепочка от точной к грубой):
    #   1) brand+model × возраст   — честнее всего: Mohave против Mohave
    #   2) brand × возраст         — если по модели данных мало
    #   3) просто возраст          — последний рубеж
    # Зачем: аномалия относительно НЕПРАВИЛЬНОЙ группы — не аномалия.
    # Kia Mohave за 15 млн на фоне Kia Rio выглядел выбросом, хотя цена
    # рыночная — группа «бренд» была неоднородной. Берём самый точный
    # уровень, где группа ещё статистически осмысленна (>= MIN_GROUP_SIZE).
    df["model_key"] = (df["brand"].fillna("") + " " + df["model"].fillna("")).str.strip()

    levels = [
        ("model", ["model_key", "age_bucket"]),
        ("brand", ["brand", "age_bucket"]),
        ("age",   ["age_bucket"]),
    ]
    df["price_z"] = np.nan
    df["z_group_level"] = ""          # на каком уровне посчитан z — для отладки
    for name, keys in levels:
        grp = df.groupby(keys, observed=True)["log_price"]
        z = grp.transform(robust_z)
        size = grp.transform("size")
        take = df["price_z"].isna() & (size >= MIN_GROUP_SIZE)
        df.loc[take, "price_z"] = z[take]
        df.loc[take, "z_group_level"] = name
    df["price_z"] = df["price_z"].fillna(0.0).round(2)

    # Асимметрия хвостов: подозрительно ДЁШЕВО — профиль приманки, идёт
    # в is_suspicious; подозрительно ДОРОГО — чаще опечатка/жадность или
    # неоднородность группы → только информационная пометка (info_flags).
    df["stat_reasons"] = ""
    df.loc[df["price_z"] < -ROBUST_Z_THRESHOLD, "stat_reasons"] = "price_anomaly_low"
    df["info_flags"] = ""
    df.loc[df["price_z"] > ROBUST_Z_THRESHOLD, "info_flags"] = "price_anomaly_high"
    return df


# ─── Шаг 5. Дубликаты-перезаливы ─────────────────────────────────────────────
def add_duplicate_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Одинаковые title+year+price при разных ad_id — вероятный перезалив.

    ДВА уровня уверенности (калибровка на реальных данных 2026-07-20:
    из 96 групп-совпадений 69 НЕ были подкреплены ничем, кроме
    title+год+цена — напр., три разных BMW X5 2016 по 17.5 млн с
    пробегами 210/241/198 тыс. — популярная модель с круглой рыночной
    ценой, совпадение, а не перезалив):
      - подтверждённый (→ is_suspicious): в группе ещё и совпал пробег
        у двух объявлений ИЛИ совпало непустое описание — та же машина;
      - неподтверждённый (→ только info_flags): совпадение может быть
        случайным, пусть решает человек при разметке, но в suspicious
        не считаем — иначе precision тонет.

    ИСКЛЮЧЕНИЕ: дилеры легитимно выкладывают несколько одинаковых НОВЫХ
    машин (пять одинаковых Changan с одной ценой — это склад, а не фрод).
    Урок: у каждого «подозрительного» паттерна ищи легитимное объяснение
    и вырезай его из правила — иначе precision утонет в дилерах."""
    df = df.copy()
    dup = df.duplicated(subset=["title", "year", "price_tenge"], keep=False)
    # NA-ловушка: у condition есть пропуски, и в pandas сравнение с NA
    # даёт не False, а NA (трёхзначная логика: True/False/Неизвестно).
    # NA в булевой маске взрывает np.where → явно решаем: «неизвестно,
    # дилер ли» трактуем как «не дилер» (fillna(False)).
    dealer_like = (df["condition"].eq("новый").fillna(False)
                   | df["labels"].fillna("").str.contains("От дилера|Новая"))
    dealer_like = dealer_like.astype(bool)

    # Подтверждение группы вторым фактором: одинаковый непустой пробег
    # или одинаковое непустое описание хотя бы у пары внутри группы.
    def corroborated(g: pd.DataFrame) -> bool:
        # Большая группа — это НЕ перезалив, а коммерческая партия
        # (реальный кейс: 24 одинаковых Chevrolet Cobalt 2023 по 4.9 млн —
        # автопарк на продаже; при 24 машинах круглые пробеги вида 33000
        # совпадают случайно — парадокс дней рождения). Настоящий
        # перезальщик дублирует объявление 2-5 раз.
        if len(g) > 5:
            return False
        # Совпавший пробег — подтверждение, только если он РЕАЛЬНЫЙ
        # (>1000 км): у дилерского стока новых машин пробеги 0/1/10/50
        # совпадают тривиально (Changan X5 2026: 0|0|0 — это склад,
        # даже если condition криво заполнен как «б/у»)
        m = g["mileage_km"].dropna()
        m = m[m > 1000]
        if m.duplicated().any():
            return True
        d = g["description"].fillna("")
        return bool(d[d != ""].duplicated().any())

    strong_groups = set()
    for key, g in df[dup].groupby(["title", "year", "price_tenge"]):
        if corroborated(g):
            strong_groups.add(key)
    keys = list(zip(df["title"], df["year"], df["price_tenge"]))
    strong = pd.Series([k in strong_groups for k in keys], index=df.index)

    df["dup_reasons"] = np.where(dup & ~dealer_like & strong,
                                 "possible_repost", "")
    weak = dup & ~dealer_like & ~strong
    df.loc[weak, "info_flags"] = (
        df.loc[weak, "info_flags"].replace("", np.nan).fillna("")
        .apply(lambda s: (s + "|" if s else "") + "repost_unconfirmed"))
    return df


# ─── Шаг 5.2. Переиспользованные фото (photo_dedup.py) ───────────────────────
def add_photo_reuse_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ворованное/переиспользованное фото — сигнал, который не видит ни
    цена, ни текст (см. photo_dedup.py). Таблица disposable (пересобирается
    там же, где clean_data) и может ещё не существовать при первом
    офлайн-прогоне — тогда просто нет флагов, а не падение."""
    df = df.copy()
    df["photo_reasons"] = ""
    engine = get_engine()
    with engine.begin() as conn:
        exists = conn.execute(text(
            "SELECT to_regclass('public.photo_duplicates')")).scalar()
        if not exists:
            return df
        dups = pd.read_sql("SELECT ad_id_a, ad_id_b FROM photo_duplicates", conn,
                            dtype={"ad_id_a": str, "ad_id_b": str})
    if dups.empty:
        return df
    flagged = set(dups["ad_id_a"]) | set(dups["ad_id_b"])
    df.loc[df["ad_id"].isin(flagged), "photo_reasons"] = "shared_photo_diff_car"
    return df


# ─── Шаг 5.5. Оправдание объяснимо дешёвых (exculpation) ─────────────────────
def exculpate(df: pd.DataFrame) -> pd.DataFrame:
    """Низкая цена бывает по двум причинам: приманка мошенника или честная
    развалюха. Если дешевизна ОБЪЯСНЕНА наблюдаемым фактором — снимаем
    подозрение, оставляя информационную пометку. Источники объяснений:
      1) слова про убитость в описании/обогащении;
      2) «Растаможен: Нет» — нерастаможенная легально стоит сильно дешевле;
      3) вердикт человека из manual_labels.csv (human-in-the-loop:
         ручная разметка встроена в пайплайн и переживает перезапуски;
         бонус — она же копит обучающую выборку для будущего классификатора).
    """
    df = df.copy()

    # Поиск с учётом отрицаний («нету никаких гнилей» — НЕ повреждение),
    # см. damage.find_damage_keywords
    has_damage = _has_damage

    # Ищем по text_full (полный комментарий продавца, если обогащён;
    # иначе огрызок листинга) ТЕКУЩИМ лексиконом — а не только по
    # сохранённому damage_keywords из enrich: тот посчитан лексиконом
    # НА МОМЕНТ обогащения и не обновляется задним числом. Поиск здесь
    # делает пополнение лексикона ретроактивным для всего уже
    # собранного текста (clean-слой на то и пересобираемый).
    explained = df["text_full"].fillna("").map(has_damage)
    if "damage_keywords" in df.columns:
        explained |= df["damage_keywords"].fillna("").str.len() > 0
    if "customs_cleared" in df.columns:
        explained |= df["customs_cleared"].eq("Нет").fillna(False)
    if "verdict" in df.columns:   # из manual_labels.csv
        explained |= df["verdict"].eq("legit").fillna(False)
    # Статус-бейдж САМОГО сайта (enrich.py): «Аварийная/Не на ходу»,
    # «Заложенная» — прямой структурный сигнал, что дешевизна законна.
    if "page_status_badge" in df.columns:
        badge = df["page_status_badge"].fillna("").str.lower()
        explained |= badge.str.contains("аварийн|не на ходу|заложен")

    mask = (df["stat_reasons"] == "price_anomaly_low") & explained
    df.loc[mask, "stat_reasons"] = ""
    df.loc[mask, "info_flags"] = (
        df.loc[mask, "info_flags"].replace("", np.nan).fillna("")
        .apply(lambda s: (s + "|" if s else "") + "low_price_explained")
    )

    # young_car_cheap — жёсткое правило (age≤4 & дёшево), но если дешевизна
    # ОБЪЯСНЕНА тем же сигналом (битая/заложена/нерастаможена/вердикт
    # legit) — оно тоже ложное. Реальный кейс: честно битый Chevrolet Onix
    # 2023 за 1.7М — сайт сам пишет «Аварийная», а правило держало его в
    # подозрительных. Снимаем токен young_car_cheap из rule_reasons (это
    # НЕ приманка), оставляя ту же пометку. Неоднозначные (молодая дешёвая
    # БЕЗ объяснения) остаются флагнутыми — их разбирает человек.
    def _drop(reasons: str, token: str) -> str:
        return "|".join(p for p in str(reasons).split("|") if p and p != token)

    # rule_reasons есть всегда в реальном пайплайне (hard-rules до exculpate);
    # guard — для юнит-тестов, вызывающих exculpate на минимальном df.
    if "rule_reasons" in df.columns:
        ycc = df["rule_reasons"].str.contains("young_car_cheap", na=False) & explained
        df.loc[ycc, "rule_reasons"] = df.loc[ycc, "rule_reasons"].apply(
            lambda s: _drop(s, "young_car_cheap"))
        # пометку добавляем, только если её ещё нет (ценовой блок выше мог
        # уже поставить low_price_explained — не задваиваем)
        need = ycc & ~df["info_flags"].fillna("").str.contains("low_price_explained")
        df.loc[need, "info_flags"] = (
            df.loc[need, "info_flags"].replace("", np.nan).fillna("")
            .apply(lambda s: (s + "|" if s else "") + "low_price_explained")
        )

    # Второй класс объяснимо низких цен — ДИЛЕРСКИЕ/КРЕДИТНЫЕ объявления:
    # в поле цены у них не рыночная цена машины, а маркетинговая
    # (первоначальный взнос, "от ...", промо рассрочки 0-0-24). Реальный
    # кейс: новый Changan Eado с z=-17 — это взнос, а не цена. Такие
    # объявления не приманка (сегмент дилеров/банков), но и цена их для
    # статистики бесполезна — снимаем флаг с отдельной пометкой.
    # ВАЖНО про отвергнутые критерии: has_monthly_price и метка
    # «первый взнос» — это один и тот же бейдж кредитного калькулятора,
    # он стоит у 68% ВСЕЙ базы (проверено: 2278/3355, распределение их
    # price_z нормальное) — как признак «цена не рыночная» он мусорный
    # и оправдал бы реальные приманки. Только явные дилерские сигналы:
    dealer_fin = (
        df["condition"].eq("новый").fillna(False)
        | df["labels"].fillna("").str.contains("Официальный дилер", case=False)
    )
    mask = (df["stat_reasons"] == "price_anomaly_low") & dealer_fin
    df.loc[mask, "stat_reasons"] = ""
    df.loc[mask, "info_flags"] = (
        df.loc[mask, "info_flags"].replace("", np.nan).fillna("")
        .apply(lambda s: (s + "|" if s else "") + "dealer_financing_price")
    )

    # Кросс-чек средней ценой kolesa (независимый эталон, см. enrich.py):
    # наш z-score считается по грубой корзине «модель × 0-3 года» и
    # мешает разные годы выпуска, из-за чего свежая машина старшего
    # модельного года ложно кажется выбросом (реальный кейс BYD Leopard 5
    # 2024 за 22.5 млн: наш z=-7.6, а kolesa — цена в пределах рынка,
    # т.к. сравнивает 2024 с 2024, а не с 2026). Если наша цена НЕ сильно
    # ниже kolesa-эталона (тот уже исключил выбросы) — наш флаг ложный,
    # снимаем. Порог 0.80: ниже 20% от рыночной kolesa — всё ещё возможна
    # честная сделка, флаг оставляем; выше — точно не приманка.
    if "kolesa_avg_price" in df.columns:
        avg = pd.to_numeric(df["kolesa_avg_price"], errors="coerce")
        # avg > 0: NaN = не обогащено, -1 = у модели нет эталона kolesa
        # (мало похожих) — в обоих случаях кросс-чек неприменим, флаг не трогаем.
        near_market = (avg > 0) & (df["price_tenge"] >= 0.80 * avg)
        mask = (df["stat_reasons"] == "price_anomaly_low") & near_market.fillna(False)
        df.loc[mask, "stat_reasons"] = ""
        df.loc[mask, "info_flags"] = (
            df.loc[mask, "info_flags"].replace("", np.nan).fillna("")
            .apply(lambda s: (s + "|" if s else "") + "kolesa_price_ok")
        )

    # Обратное усиление: аномально дёшево + давление СРОЧНОСТЬЮ.
    # «Срочно» само по себе безобидно (люди правда спешат), но в паре
    # с глубоким дисконтом — профиль приманки: скидка отключает разум,
    # срочность не даёт ему включиться. Правила-КОМБИНАЦИИ точнее
    # правил-одиночек: precision растёт за счёт пересечения условий.
    urgency = pd.Series(False, index=df.index)
    for col in ["description", "seller_comment"]:
        if col in df.columns:
            urgency |= df[col].fillna("").str.lower().str.contains("срочн")
    cheap_urgent = (df["stat_reasons"] == "price_anomaly_low") & urgency
    df.loc[cheap_urgent, "stat_reasons"] = "price_anomaly_low|cheap_and_urgent"

    # Вердикт «fraud» от человека — наоборот, флаг намертво
    if "verdict" in df.columns:
        fraud = df["verdict"].eq("fraud").fillna(False)
        df.loc[fraud & (df["stat_reasons"] == ""), "stat_reasons"] = \
            "confirmed_by_review"
    return df



def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def join_reasons(row):
        parts = [row["rule_reasons"], row["stat_reasons"], row["dup_reasons"],
                 row["photo_reasons"]]
        return "|".join(p for p in parts if p)

    df["suspicion_reasons"] = df.apply(join_reasons, axis=1)
    df["is_suspicious"] = (df["suspicion_reasons"] != "").astype(int)
    return df.drop(columns=["rule_reasons", "stat_reasons", "dup_reasons",
                             "photo_reasons"])


def quality_report(df: pd.DataFrame):
    """Data quality report: первое, что показывают заказчику/тимлиду."""
    log.info("=" * 60)
    log.info(f"Строк: {len(df)}, подозрительных: {df['is_suspicious'].sum()} "
             f"({df['is_suspicious'].mean():.1%})")
    log.info("-" * 60)
    all_reasons = df.loc[df["is_suspicious"] == 1, "suspicion_reasons"] \
                    .str.split("|").explode().value_counts()
    for reason, cnt in all_reasons.items():
        log.info(f"  {reason:<25} {cnt}")
    info = df["info_flags"].replace("", np.nan).dropna().value_counts()
    if len(info):
        log.info("Информационные пометки (не считаются подозрительными):")
        for reason, cnt in info.items():
            log.info(f"  {reason:<25} {cnt}")
    log.info("-" * 60)
    log.info("Пропуски (top):")
    na = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    for col, pct in na[na > 0].head(6).items():
        log.info(f"  {col:<20} {pct}%")
    log.info("=" * 60)


def main():
    engine = get_engine()
    # raw_ads/enriched — PK по ad_id уже гарантирует уникальность в
    # Postgres, дедуп на чтении (как раньше для CSV) больше не нужен.
    df = pd.read_sql("SELECT * FROM raw_ads", engine, dtype={"ad_id": str})

    # Обогащение (Job 1b), если уже собрано: растаможка, привод, руль,
    # цвет, слова про состояние + ДОЗАПОЛНЕНИЕ пробега со страницы
    enr = pd.read_sql("SELECT * FROM enriched", engine, dtype={"ad_id": str})
    if not enr.empty:
        cols = ["ad_id", "customs_cleared", "drive", "steering", "color",
                "generation", "page_mileage_km", "damage_keywords",
                "seller_comment", "kolesa_avg_price", "page_status_badge"]
        df = df.merge(enr[[c for c in cols if c in enr.columns]],
                      on="ad_id", how="left")
        filled = df["mileage_km"].isna() & df["page_mileage_km"].notna()
        df.loc[filled, "mileage_km"] = df.loc[filled, "page_mileage_km"]
        log.info(f"Обогащено: {enr.shape[0]}, пробегов дозаполнено: {filled.sum()}")

    # COALESCE текстов: единая колонка «лучший доступный текст».
    # seller_comment (полный, со страницы) > description (огрызок из листинга).
    # Coalesce — приём из SQL: «первое непустое из списка источников».
    # Для NLP и ручного разбора смотри text_full, а не description.
    df["text_full"] = df.get("seller_comment", pd.Series(index=df.index, dtype="object")) \
                        .fillna(df["description"])

    # Ручные вердикты (ad_id, verdict[legit|fraud|unknown], comment)
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str}) \
                .drop_duplicates("ad_id", keep="last")
        df = df.merge(lab[["ad_id", "verdict"]], on="ad_id", how="left")
        log.info(f"Ручных вердиктов: {lab.shape[0]}")

    df = normalize(df)
    df = add_missing_indicators(df)
    df = apply_hard_rules(df)
    df = add_price_outliers(df)
    df = exculpate(df)
    df = add_duplicate_flags(df)
    df = add_photo_reuse_flags(df)
    df = finalize(df)

    # Подмешиваем статусы, если Job 1c уже что-то собрал (однострочный
    # upsert по ad_id в Postgres — дедуп на чтении тут не нужен)
    st = pd.read_sql("SELECT ad_id, status FROM ad_status", engine, dtype={"ad_id": str})
    if not st.empty:
        df = df.merge(st, on="ad_id", how="left")
        df["status"] = df["status"].fillna("active")
    else:
        df["status"] = "active"

    df.to_csv(OUT_CSV, index=False)
    # disposable: пересобирается из raw каждый прогон, поэтому TRUNCATE+
    # заливка внутри одной транзакции, а не ручной upsert по ключу
    with engine.begin() as conn:
        df.to_sql("clean_data", conn, if_exists="replace", index=False)
    quality_report(df)
    log.info(f"Сохранено → {OUT_CSV} и в таблицу clean_data")


if __name__ == "__main__":
    main()