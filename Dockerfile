FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hawk-derm
COPY requirements-lock.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-lock.txt && pip install --no-cache-dir .

COPY . .
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
