# planer-app: Python 3.11 + PyMongo, RapidFuzz, scikit-learn, sentence-transformers, Streamlit.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY experiments ./experiments
RUN pip install --no-deps -e .

# Download the embedding model at build time so the container works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
