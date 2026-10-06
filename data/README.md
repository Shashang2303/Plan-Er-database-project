# Benchmark Data Directory

Raw benchmark datasets are downloaded and unzipped into `data/raw/` automatically by the PlanER ingestion pipeline:

```bash
planer ingest all
```

Supported benchmarks:
- `Abt-Buy`
- `Amazon-GoogleProducts`
- `DBLP-ACM`
- `DBLP-Scholar`

Raw CSV archives are excluded from git tracking via `.gitignore`.
