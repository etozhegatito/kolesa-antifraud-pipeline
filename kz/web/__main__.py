# -*- coding: utf-8 -*-
"""Запуск веб-интерфейса: python -m kz.web

Слушаем только 127.0.0.1 — приложение локальное, в нём нет ни аутентификации,
ни ограничения частоты запросов, наружу его открывать нельзя.
"""

import sys

import uvicorn

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main():
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"Открой: http://{HOST}:{port}")
    uvicorn.run("kz.web.app:app", host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
