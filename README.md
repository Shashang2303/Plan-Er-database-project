# PlanER: Cost-Based Optimization of Hybrid Classical–LLM Entity Resolution Plans in MongoDB

**Course:** COMP 8157 Advanced Database Topics (Fall 2026)  
**Authors:**
- **Shashang Patel** ([@shashang2303](https://github.com/shashang2303)) – `patel3cd@uwindsor.ca` / `shashangpatel6@gmail.com`
- **Nauman Syed** ([@naumanizami](https://github.com/naumanizami)) – `syed9i@uwindsor.ca` / `nauman2046@gmail.com`

---

## 📌 Project Overview

**PlanER** treats entity resolution (ER) as a declarative request compiled into a physical plan executed directly inside MongoDB. Rather than naively invoking expensive Large Language Models (LLMs) on every candidate pair, PlanER combines:
1. **Four Index-Backed Blocking Access Paths** (B-tree key, multikey rare token, BM25 `$search`, and HNSW vector `$vectorSearch`).
2. **A Heterogeneous Matcher Cascade** across classical rules (RapidFuzz), probabilistic record linkage (Fellegi–Sunter fitted by EM), supervised machine learning (Random Forest), and local LLMs (Llama 3.1 8B, Qwen 2.5 7B via Ollama).
3. **Deterministic Verdict Caching (`llm_cache`)** ensuring model calls are never repeated.
4. **Cost-Based Optimizer** that uses stratified sampling to predict candidate escalation rates, monetary/compute cost, and F1 scores to select the Pareto-optimal plan under budget or quality constraints.

---

## 🏗️ Repository Structure

```text
├── docker/                 # Docker Compose & service configs (MongoDB 8.2, mongot, Ollama)
├── docs/                   # Course deliverables (D.1, D.2.1, D.2.2, D.4.1, D.4.2, D.3.1)
├── data/                   # Benchmark datasets registry & mongoimport staging
├── src/planer/             # Core Python package
│   ├── blocking/           # Access paths: B-Tree, Multikey, $search, $vectorSearch
│   ├── matchers/           # Rule, Fellegi-Sunter (EM), Random Forest, and Local LLM
│   ├── optimizer/          # Stratified sampling, cost & F1 estimation, plan pruning
│   ├── cascade.py          # Matcher cascade engine with escalation
│   ├── clustering.py       # Connected components & transitivity analysis
│   ├── ingest.py           # Dataset downloader, normalizer, and embedder
│   ├── request.py          # Declarative ER request and Physical Plan definitions
│   └── cli.py              # CLI entrypoint (`planer <command>`)
├── ui/                     # Interactive Streamlit application
├── experiments/            # Profiling and evaluation experiments
└── tests/                  # Unit tests for blocking, matchers, cascade, and optimizer
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shashang2303/Plan-Er-database-project.git
cd Plan-Er-database-project

# Install dependencies
pip install -e .
cp .env.example .env
```

### 2. Start Services via Docker
```bash
docker compose -f docker/docker-compose.yml up -d
```

### 3. Ingest Benchmarks
```bash
planer status
planer ingest all
```

### 4. Run the Streamlit Interface
```bash
streamlit run ui/app.py
```

---

## 📊 Benchmark Datasets
- **DBLP-ACM** (clean bibliographic linkage)
- **Abt-Buy** (noisy product titles & attributes)
- **Amazon-GoogleProducts** (heterogeneous e-commerce catalog)
- **DBLP-Scholar** (~168M candidate pairs, stress-testing blocking scale)
