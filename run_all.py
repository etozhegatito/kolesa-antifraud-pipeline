# -*- coding: utf-8 -*-
"""
run_all.py — мини-оркестратор: запускает джобы в ПРАВИЛЬНОМ порядке.

Зачем он нужен (проблема из ревью): clean_data.csv собирался ДО обогащения
и артефакты устаревали. Ручной запуск джобов в голове держать нельзя —
порядок должен быть закодирован. Это простейшая форма оркестрации
(в проде это Airflow/Prefect, у нас — 60 строк на subprocess).

Порядок (DAG — directed acyclic graph, граф зависимостей):
  parser.py         листинг: паспорта, sightings, фото
  check_status.py   статусы исчезнувших (archived/deleted)
  clean.py  #1      первичная чистка → очередь приоритетов для enrich
  enrich.py      ┐  обогащение страниц (подозрительные первыми)
  photo_dedup.py ┘  ПАРАЛЛЕЛЬНО: pHash фоток («одно фото, разные машины»)
  clean.py  #2      финальная чистка УЖЕ с обогащением и фото-флагами
  explore.py        отчёт, дашборд, очередь на разметку

Почему enrich и photo_dedup параллельно, и почему это НЕ ломает
вежливость к сайту: вежливость — она per-host (частота запросов к
одному хосту), а не per-компьютер. enrich стучится на kolesa.kz,
photo_dedup — на CDN картинок (kcdn.kz), это разные хосты с разными
лимитами. Частота запросов к каждому в отдельности не меняется —
а стены-времени экономится ~12 минут (photo_dedup целиком прячется
внутри более длинного enrich). Параллелить два джоба на ОДИН хост
(например, parser + enrich) — нельзя, это удвоило бы частоту.

Запуск: python run_all.py            (весь пайплайн)
        python run_all.py --light    (только сбор нового листинга + пересборка
                                      флагов; per-ad сеть — статусы/обогащение/
                                      фото — НЕ трогаем, их отдаём бюджетному
                                      catch_up. Легче и быстрее; во время
                                      backfill не долбит kolesa мимо бюджета)
        python run_all.py --fast     (совсем без сети — только пересборка
                                      clean+EDA из уже собранного raw)
"""

# ─── Самопроверка файла (защита от путаницы при копировании) ────────────────
# Если этот код оказался в файле с другим именем — останавливаемся сразу,
# а не делаем «не то» молча. Тихая подмена хуже громкого падения.
import pathlib as _p
_expected = "run_all.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ОШИБКА: этот код — {_expected}, а файл называется "
        f"{_p.Path(__file__).name}. Файлы перепутаны при копировании!")


import subprocess
import sys
import time

STEP_PARSER  = ("Job 1  · листинг",         [sys.executable, "parser.py"])
STEP_STATUS  = ("Job 1c · статусы",         [sys.executable, "check_status.py"])
STEP_CLEAN   = ("Job 2  · чистка",          [sys.executable, "clean.py"])
STEP_ENRICH  = ("Job 1b · обогащение",      [sys.executable, "enrich.py"])
STEP_PHOTOS  = ("Job 1d · фото-дедуп",      [sys.executable, "photo_dedup.py"])
STEP_EXPLORE = ("Job 3  · EDA/отчёт",       [sys.executable, "explore.py"])


def run_step(step) -> None:
    """Последовательный шаг с fail-fast."""
    name, cmd = step
    print(f"\n{'═'*60}\n▶ {name}\n{'═'*60}")
    t = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"  … {time.time()-t:.0f}s, код {rc}")
    if rc != 0:
        # Fail fast: упавший шаг делает следующие бессмысленными
        # (clean без свежего raw, explore без свежего clean).
        print(f"✖ Шаг «{name}» упал — останавливаем пайплайн.")
        sys.exit(rc)


def run_parallel(step_a, step_b) -> None:
    """Два шага одновременно (РАЗНЫЕ хосты — см. докстринг модуля).
    Ждём обоих до конца, даже если один упал: второй-то работает и
    его результат резюмируемый — обрывать его на середине глупо.
    Потом fail-fast, если хоть один вернул ошибку."""
    (name_a, cmd_a), (name_b, cmd_b) = step_a, step_b
    print(f"\n{'═'*60}\n▶ {name_a}  ∥  {name_b}  (параллельно)\n{'═'*60}")
    t = time.time()
    proc_a = subprocess.Popen(cmd_a)
    proc_b = subprocess.Popen(cmd_b)
    rc_a, rc_b = proc_a.wait(), proc_b.wait()
    print(f"  … {time.time()-t:.0f}s, коды {rc_a}/{rc_b}")
    for name, rc in ((name_a, rc_a), (name_b, rc_b)):
        if rc != 0:
            print(f"✖ Шаг «{name}» упал — останавливаем пайплайн.")
            sys.exit(rc)


def main():
    t0 = time.time()
    fast  = "--fast" in sys.argv    # совсем без сети (fast «сильнее» light)
    light = "--light" in sys.argv   # только новый листинг, без per-ad сети

    if not fast:
        run_step(STEP_PARSER)                      # свежий листинг (новьё)
    if not fast and not light:
        # Тяжёлые per-ad сетевые джобы. В режиме --light их пропускаем и
        # отдаём бюджетному catch_up (см. README: иначе run_all грузил бы
        # kolesa мимо суточного бюджета catch_up — двойная нагрузка на IP).
        run_step(STEP_STATUS)
        run_step(STEP_CLEAN)                       # пасс 1: очередь для enrich
        run_parallel(STEP_ENRICH, STEP_PHOTOS)     # kolesa.kz ∥ kcdn.kz

    run_step(STEP_CLEAN)                           # финальная чистка (пасс 2)
    run_step(STEP_EXPLORE)
    mode = "--fast" if fast else ("--light" if light else "полный")
    print(f"\n✔ Пайплайн ({mode}) завершён за {(time.time()-t0)/60:.1f} мин")


if __name__ == "__main__":
    main()
