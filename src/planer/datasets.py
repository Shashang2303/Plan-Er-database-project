"""Benchmark registry (FR-23): every dataset pair is described by configuration only.

The four benchmarks come from the Leipzig entity-resolution benchmark collection
(Koepcke, Thor and Rahm, PVLDB 2010) and include complete ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LEIPZIG_BASE_URL = "https://dbs.uni-leipzig.de/files/datasets"


@dataclass(frozen=True)
class SourceSpec:
    name: str           # MongoDB collection name, e.g. "abt"
    file: str           # CSV file inside the dataset folder
    title_attr: str     # attribute that becomes title_norm / tokens / embedding


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    zip_name: str
    a: SourceSpec
    b: SourceSpec
    gold_file: str
    gold_cols: tuple[str, str]
    # attribute mapping a_attr -> b_attr used by the matchers (FR-1)
    attr_map: dict[str, str]
    kind: str                       # "product" or "bibliographic": selects blocking-key rule
    numeric_attrs: tuple[str, ...] = ()
    # blocking-key rule for the B-tree path (normalize.BKEY_RULES), chosen from measured pair
    # completeness: see `planer bkeys <dataset>`
    bkey_rule: str = "first_token"
    # multikey path: a token is "rare" if its document frequency <= this fraction of all records
    multikey_max_df_frac: float = 0.01
    description: str = ""
    default_k: int = 10             # default top-k for the $search / $vectorSearch paths
    extra: dict = field(default_factory=dict)

    @property
    def folder(self) -> str:
        return self.zip_name.removesuffix(".zip")

    @property
    def sources(self) -> tuple[str, str]:
        return (self.a.name, self.b.name)


DATASETS: dict[str, DatasetSpec] = {
    "dblp_acm": DatasetSpec(
        key="dblp_acm",
        zip_name="DBLP-ACM.zip",
        a=SourceSpec("dblp2", "DBLP2.csv", "title"),
        b=SourceSpec("acm", "ACM.csv", "title"),
        gold_file="DBLP-ACM_perfectMapping.csv",
        gold_cols=("idDBLP", "idACM"),
        attr_map={"title": "title", "authors": "authors", "venue": "venue", "year": "year"},
        kind="bibliographic",
        bkey_rule="title_prefix",
        multikey_max_df_frac=0.005,
        numeric_attrs=("year",),
        description="Clean bibliographic data, ~2.6K x 2.3K records",
    ),
    "abt_buy": DatasetSpec(
        key="abt_buy",
        zip_name="Abt-Buy.zip",
        a=SourceSpec("abt", "Abt.csv", "name"),
        b=SourceSpec("buy", "Buy.csv", "name"),
        gold_file="abt_buy_perfectMapping.csv",
        gold_cols=("idAbt", "idBuy"),
        attr_map={"name": "name", "description": "description", "price": "price"},
        kind="product",
        bkey_rule="first_token",
        multikey_max_df_frac=0.02,
        numeric_attrs=("price",),
        description="Noisy product data, ~1.1K records per side",
    ),
    "amazon_google": DatasetSpec(
        key="amazon_google",
        zip_name="Amazon-GoogleProducts.zip",
        a=SourceSpec("amazon", "Amazon.csv", "title"),
        b=SourceSpec("google", "GoogleProducts.csv", "name"),
        gold_file="Amzon_GoogleProducts_perfectMapping.csv",  # (sic) file name in the archive
        gold_cols=("idAmazon", "idGoogleBase"),
        attr_map={"title": "name", "manufacturer": "manufacturer", "price": "price", "description": "description"},
        kind="product",
        bkey_rule="first_token",
        multikey_max_df_frac=0.02,
        numeric_attrs=("price",),
        description="Noisy software products, ~1.4K x 3.2K records",
    ),
    "dblp_scholar": DatasetSpec(
        key="dblp_scholar",
        zip_name="DBLP-Scholar.zip",
        a=SourceSpec("dblp1", "DBLP1.csv", "title"),
        b=SourceSpec("scholar", "Scholar.csv", "title"),
        gold_file="DBLP-Scholar_perfectMapping.csv",
        gold_cols=("idDBLP", "idScholar"),
        attr_map={"title": "title", "authors": "authors", "venue": "venue", "year": "year"},
        kind="bibliographic",
        bkey_rule="title_prefix",
        multikey_max_df_frac=0.001,
        numeric_attrs=("year",),
        description="Blocking at scale: ~2.6K x 64K records (~168M possible pairs)",
    ),
}


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}") from None


def dataset_for_sources(sources: list[str] | tuple[str, str]) -> DatasetSpec:
    for spec in DATASETS.values():
        if tuple(sources) == spec.sources:
            return spec
    raise KeyError(f"no dataset registered for sources {sources!r}")
