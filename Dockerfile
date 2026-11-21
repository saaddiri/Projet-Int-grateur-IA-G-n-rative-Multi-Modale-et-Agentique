# Image unique de MaintiDoc : backend FastAPI, bot Telegram, interface Streamlit.
# Les modèles Ollama tournent hors de l'image (service ollama ou Ollama de l'hôte) ;
# les modèles Hugging Face sont mis en cache dans un volume.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# PyTorch CPU (version figée), puis dépendances figées du projet
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY maintidoc ./maintidoc
COPY scripts ./scripts
COPY app.py versions.lock.json ./
COPY data ./data

# Utilisateur non privilégié ; données et caches dans des volumes
RUN useradd --create-home --uid 1000 maintidoc \
    && mkdir -p /app/storage /app/uploads /app/models /cache/huggingface \
    && chown -R maintidoc:maintidoc /app /cache
USER maintidoc

EXPOSE 8000 8501
CMD ["uvicorn", "maintidoc.api:app", "--host", "0.0.0.0", "--port", "8000"]
