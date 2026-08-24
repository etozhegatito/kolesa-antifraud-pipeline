# -*- coding: utf-8 -*-
"""Локальный сервер разметки.

Нужен потому, что страница, открытая как file://, писать на диск не может в
принципе, а без записи вердикты приходилось переносить копипастой. Слушает
только 127.0.0.1: инструмент локальный, наружу его открывать незачем.
"""

import json

from kz.report.label_cards.journal import LABELS_CSV, upsert_verdict

def serve(html: str, facts: dict, port: int = 8765, on_ready=None) -> None:
    """Локальный сервер: отдаёт карточки и дописывает вердикты в журнал.

    Нужен потому, что страница, открытая как file://, писать на диск не может
    в принципе — а без записи выборы приходилось переносить копипастой.
    Слушаем только 127.0.0.1: инструмент локальный, наружу его открывать
    незачем. Пишем строго через append_verdict (валидация + append-only).

    port=0 означает «любой свободный», а on_ready(port) вызывается сразу
    после привязки. Это ради тестов: с жёстким портом два одновременных
    прогона дрались за него, а фиксированная пауза «подождём, наверное
    поднялся» превращала тест в лотерею на медленной машине.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import unquote

    page = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if path.startswith("/photos/"):
                # Скачанные картинки. Путь нормализуем и проверяем, что он не
                # вылезает за каталог фотографий: иначе через ../ можно было бы
                # прочитать любой файл на диске.
                from kz.collect.photo_fetch import PHOTO_DIR
                rel = unquote(path[len("/photos/"):])
                target = (PHOTO_DIR / rel).resolve()
                if PHOTO_DIR.resolve() in target.parents and target.is_file():
                    return self._send(200, target.read_bytes(), "image/jpeg")
                return self._send(404, b"not found", "text/plain")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/verdict":
                return self._send(404, b'{"error":"not found"}',
                                  "application/json")
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
                ad_id = str(data.get("ad_id", ""))
                # ad_id принимаем только из числа показанных карточек:
                # запись в журнал не должна зависеть от того, что пришло в теле
                if ad_id not in facts:
                    raise ValueError(f"неизвестный ad_id: {ad_id!r}")
                upsert_verdict(ad_id, str(data.get("verdict", "")),
                               str(data.get("comment", "")), facts[ad_id])
            except Exception as e:                  # noqa: BLE001 — ответ клиенту
                return self._send(400, json.dumps({"error": str(e)},
                                  ensure_ascii=False).encode(),
                                  "application/json; charset=utf-8")
            self._send(200, b'{"ok":true}', "application/json")

        def log_message(self, *a):                  # тише в консоли
            pass

    srv = HTTPServer(("127.0.0.1", port), Handler)
    port = srv.server_address[1]        # при port=0 настоящий выбрала ОС
    if on_ready:
        on_ready(port)
    print(f"\nОткрой: http://127.0.0.1:{port}")
    print(f"Вердикты дописываются в {LABELS_CSV} сразу при нажатии.")
    print("Остановить: Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        srv.server_close()
