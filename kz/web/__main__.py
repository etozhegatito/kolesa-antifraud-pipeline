# -*- coding: utf-8 -*-
"""Start the web interface with ``python -m kz.web``.

The local entry point binds to 127.0.0.1. Manual labelling has no
authentication and must never be exposed directly to the internet.
"""

import sys

import uvicorn

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main():
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"Open: http://{HOST}:{port}")
    uvicorn.run("kz.web.app:app", host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
