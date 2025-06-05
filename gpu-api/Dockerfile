FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/* && pip install --upgrade pip && pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv pip compile pyproject.toml -o requirements.txt && pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=3s CMD wget --quiet --tries=1 --spider http://127.0.0.1:8888/health/ || exit 1

CMD ["python", "-m", "app"]
