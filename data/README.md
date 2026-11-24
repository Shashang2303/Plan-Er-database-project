# Benchmark Data Directory

`data/raw/` holds the four benchmark dataset pairs used by PlanER, as published in the
Leipzig entity-resolution benchmark collection (Köpcke, Thor and Rahm, PVLDB 2010):

- Source: https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
- Files: https://dbs.uni-leipzig.de/files/datasets/<name>.zip

| Folder | Source A | Source B | Gold matches (`*_perfectMapping.csv`) |
|---|---|---|---|
| `DBLP-ACM` | DBLP2.csv (2,616) | ACM.csv (2,294) | 2,224 |
| `Abt-Buy` | Abt.csv (1,081) | Buy.csv (1,092) | 1,097 |
| `Amazon-GoogleProducts` | Amazon.csv (1,363) | GoogleProducts.csv (3,226) | 1,300 |
| `DBLP-Scholar` | DBLP1.csv (2,616) | Scholar.csv (64,263) | 5,347 |

The CSVs are committed so the repository is self-contained. If they are missing,
`planer ingest all` downloads and unzips them again, then normalizes, embeds and loads
them into MongoDB.
