import os

host = os.getenv("HOST", "0.0.0.0")  # noqa: S104
port = os.getenv("PORT", "8880")
bind = f"{host}:{port}"

workers = 1

worker_class = "uvicorn.workers.UvicornWorker"

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
