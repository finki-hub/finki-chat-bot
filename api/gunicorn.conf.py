import os

host = os.getenv("HOST", "0.0.0.0")  # noqa: S104
port = os.getenv("PORT", "8880")
bind = f"{host}:{port}"

workers = int(os.getenv("WORKERS", "4"))

worker_class = "uvicorn.workers.UvicornWorker"

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
