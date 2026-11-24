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

## 🔗 Project Links & Deliverables

| | |
|---|---|
| **Code repository** | https://github.com/Shashang2303/Plan-Er-database-project |
| **Project management (Hive)** | https://app.hive.com/workspace/Z5kWMP9jXME8StoTL?projectId=vq8eTbJ9SWJFRMZSA |
| **Running application** | Self-hosted: `docker compose -f docker/docker-compose.yml up -d`, then `streamlit run ui/app.py` → http://localhost:8501 |

| Deliverable | File |
|---|---|
| D.1 Pre-proposal | [docs/D1_Preproposal_Team07.pdf](docs/D1_Preproposal_Team07.pdf) |
| D.2.1 Full proposal | [docs/D2_1_Full_Proposal_Team07.pdf](docs/D2_1_Full_Proposal_Team07.pdf) |
| D.2.2 Proposal presentation | [docs/PlanER_Proposal_Presentation.pptx](docs/PlanER_Proposal_Presentation.pptx) |
| D.3.1 Midterm status report | [docs/PlanER_D3_1_Midterm_Status_Report_Team07_v1.1.pdf](docs/PlanER_D3_1_Midterm_Status_Report_Team07_v1.1.pdf) |
| D.4.1 User requirements & analysis | [docs/D4_1_User_Requirements_Analysis_Team07.pdf](docs/D4_1_User_Requirements_Analysis_Team07.pdf) |
| D.4.2 Design document | [docs/PlanER_D4_2_Design_Document_Team07.pdf](docs/PlanER_D4_2_Design_Document_Team07.pdf) |
| D.4.3 Project full application | [docs/PlanER_D4_3_Project_Full_Application_Team07.pdf](docs/PlanER_D4_3_Project_Full_Application_Team07.pdf) |
| D.4.4 Deployment document | [docs/PlanER_D4_4_Deployment_Document_Team07.pdf](docs/PlanER_D4_4_Deployment_Document_Team07.pdf) |
| D.4.5 User guide | [docs/PlanER_D4_5_User_Guide_Team07.pdf](docs/PlanER_D4_5_User_Guide_Team07.pdf) |

Datasets: [data/README.md](data/README.md) (Leipzig ER benchmarks, committed under `data/raw/`).

---

## 🏗️ Repository Structure

```text
├── docker/                 # Docker Compose & service configs (MongoDB 8.2, mongot, Ollama)
├── docs/                   # Course deliverables (D.1 – D.4.5)
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
planer ingest all          # download, normalize, embed and load (MongoDB on localhost:27018)
planer index all           # $search and $vectorSearch indexes on mongot
planer llm-pull            # local LLM into Ollama (default qwen2.5:1.5b)
```

### 4. Profile, Optimize, Execute, Evaluate
```bash
planer profile-blocking abt_buy              # pair completeness, latency, explain() of every access path
planer profile-matchers abt_buy              # stratified labeled sample + per-matcher cost and quality
planer optimize abt_buy --theta 0.85         # cheapest plan predicted to reach F1 >= 0.85 (Pareto set)
planer optimize abt_buy --budget 10 --execute  # best plan within $10, run it, log predicted vs observed
planer errors                                # optimizer prediction error per run (FR-20)
planer evaluate abt_buy                      # PlanER vs LLM-only, best classical, embedding proxy
```
The optimizer skips plans expected to make more than `LLM_MAX_CALLS_PER_RUN` (default 300) new LLM
calls, because a CPU-only local model takes a few seconds per pair.

### 5. Run the Streamlit Interface
```bash
streamlit run ui/app.py
```

---

## 📊 Benchmark Datasets
- **DBLP-ACM** (clean bibliographic linkage)
- **Abt-Buy** (noisy product titles & attributes)
- **Amazon-GoogleProducts** (heterogeneous e-commerce catalog)
- **DBLP-Scholar** (~168M candidate pairs, stress-testing blocking scale)

