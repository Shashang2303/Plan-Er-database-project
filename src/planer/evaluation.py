"""T19 / FR-22, AC-3, AC-9: PlanER against three baselines on the same held-out test split.

All four methods share the blocking candidates of the plan PlanER selected, so the comparison
isolates matching. Quality is measured on the `test` split only.

  PlanER           the optimizer's plan, executed end to end (observed)
  best classical   the single classical matcher (rules, fs or rf) with the best F1 on the
                   *estimation* sample, executed end to end (observed)
  LLM-only         every candidate pair goes to the LLM. The call count and cost are exact (N1 calls);
                   precision and recall are estimated from a random LLM-scored sample of test pairs,
                   because scoring all of them would take hours on CPU.
  embedding proxy  LOTUS-style cascade: the embedding cosine decides confident pairs, thresholds chosen
                   on the estimation sample; uncertain pairs go to the LLM (same sampling rule).
"""
from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bson import ObjectId
from pymongo.database import Database

from . import db as dbm
from .config import PROJECT_ROOT, settings
from .datasets import get_dataset
from .executor import execute
from .matchers import PairBatch
from .matchers.llm import OllamaMatcher
from .metrics import in_split, load_gold, pair_completeness
from .optimizer import latest_profile, optimize_and_execute
from .optimizer.profiling import summarize
from .request import Plan, default_request, submit

log = logging.getLogger(__name__)

PROXY_PRECISION = 0.90      # proxy may auto-accept only where its sample precision is at least this
PROXY_RECALL_LOSS = 0.10    # and may auto-reject at most this share of the sample's true matches


def _prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def llm_expected(llm: OllamaMatcher, batch: PairBatch, idx: np.ndarray, labels: np.ndarray, budget: int,
                 seed: int = 7) -> tuple[float, float, int]:
    """Expected (TP, FP) if the LLM judged every pair in idx; scores at most `budget` of them."""
    if len(idx) == 0:
        return 0.0, 0.0, 0
    rng = np.random.default_rng(seed)
    scored = idx if len(idx) <= budget else rng.choice(idx, size=budget, replace=False)
    verdict = llm.score(batch.subset(scored)) >= 0.5
    scale = len(idx) / len(scored)
    y = labels[scored]
    return float((verdict & y).sum() * scale), float((verdict & ~y).sum() * scale), len(scored)


def proxy_thresholds(prof) -> tuple[float, float]:
    """Choose (l, u) for the embedding proxy on the weighted estimation sample (no test labels)."""
    s = prof.sample
    spec = get_dataset(prof.dataset)
    batch = PairBatch.load(dbm.get_db(), spec, s.pairs)
    cos = batch.features().col("emb_cos")
    cos = np.nan_to_num(cos, nan=0.0)
    w, y = s.weight, s.labels
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    upper = 1.0
    for u in grid:                         # lowest u whose auto-accepted pairs are precise enough
        sel = cos >= u
        if sel.any() and (w[sel] * y[sel]).sum() / w[sel].sum() >= PROXY_PRECISION:
            upper = float(u)
            break
    lower = 0.0
    total_pos = (w * y).sum()
    for lo in grid[::-1]:                  # highest l that drops few true matches
        lost = (w[cos <= lo] * y[cos <= lo]).sum()
        if total_pos and lost / total_pos <= PROXY_RECALL_LOSS and lo < upper:
            lower = float(lo)
            break
    return lower, upper


def evaluate(dataset: str, objective: dict | None = None, llm_budget: int = 100,
             db: Database | None = None) -> list[dict]:
    db = db if db is not None else dbm.get_db()
    t0 = time.perf_counter()
    spec = get_dataset(dataset)
    prof = latest_profile(dataset, db)
    rate, label_price = settings.usd_per_compute_second, settings.usd_per_label
    rows: list[dict] = []

    # 1. PlanER ---------------------------------------------------------------------------------
    req_id = submit(default_request(dataset, objective or {"type": "theta", "value": 0.85}), db)
    res, rep = optimize_and_execute(req_id, db)
    plan = res.chosen.plan
    rows.append({"method": "PlanER", "plan": plan.describe(), "precision": rep.precision, "recall": rep.recall,
                 "f1": rep.f1_pair, "llm_calls": rep.llm_calls + rep.cache_hits, "cost_usd": rep.cost_usd,
                 "estimated": False, "pred_cost_usd": res.chosen.cost, "pred_f1": res.chosen.f1})

    # shared candidates and test split -----------------------------------------------------------
    cands = [(c["a_id"], c["b_id"]) for c in db[dbm.CANDIDATES].find({"req_id": ObjectId(rep.req_id)}, {"a_id": 1, "b_id": 1})]
    n1 = len(cands)
    gold = load_gold(dataset, db)
    gold_test = {p for p in gold if in_split(p[0], "test")}
    test_pairs = [p for p in cands if in_split(p[0], "test")]
    pc_test = pair_completeness(test_pairs, gold_test)
    batch = PairBatch.load(db, spec, test_pairs)
    labels = batch.labels(gold)
    missed = len(gold_test) - int(labels.sum())        # matches blocking never produced
    block_cost = rep.block_cost_usd

    # 2. best single classical matcher, chosen on the estimation sample -------------------------
    summ = summarize(prof)
    classical = [m for m in ("rules", "fs", "rf") if m in summ]
    best = max(classical, key=lambda m: summ[m]["f1"])
    req_b = submit(default_request(dataset, objective), db)
    rep_b = execute(req_b, Plan(plan.access_path, [best], k=plan.k), db=db)
    rows.append({"method": f"Best classical ({best})", "plan": f"{plan.access_path} | {best}[final]",
                 "precision": rep_b.precision, "recall": rep_b.recall, "f1": rep_b.f1_pair, "llm_calls": 0,
                 "cost_usd": rep_b.cost_usd, "estimated": False})

    llm = OllamaMatcher()
    c_llm = prof.matchers["llm"].cost_per_pair_usd if "llm" in prof.matchers else 4.0 * rate

    # 3. LLM-only ---------------------------------------------------------------------------------
    all_idx = np.arange(len(batch))
    tp, fp, n_scored = llm_expected(llm, batch, all_idx, labels, llm_budget)
    q = _prf(tp, fp, labels.sum() - tp + missed)
    rows.append({"method": "LLM-only", "plan": f"{plan.access_path} | {llm.name}[final]", **q, "llm_calls": n1,
                 "cost_usd": block_cost + n1 * c_llm, "estimated": True,
                 "note": f"P/R from {n_scored} LLM-scored test pairs"})

    # 4. LOTUS-style embedding-proxy cascade ------------------------------------------------------
    lo, hi = proxy_thresholds(prof)
    cos = np.nan_to_num(batch.features().col("emb_cos"), nan=0.0)
    auto_match = cos >= hi
    escalate = np.where((cos > lo) & (cos < hi))[0]
    tp_auto = float((auto_match & labels).sum())
    fp_auto = float((auto_match & ~labels).sum())
    tp_l, fp_l, n_scored = llm_expected(llm, batch, escalate, labels, llm_budget, seed=11)
    tp, fp = tp_auto + tp_l, fp_auto + fp_l
    q = _prf(tp, fp, labels.sum() - tp + missed)
    esc_share = len(escalate) / len(batch) if len(batch) else 0.0
    calls = int(round(esc_share * n1))                  # escalation over all candidates, not only test
    rows.append({"method": "Embedding proxy (LOTUS-style)", "plan": f"{plan.access_path} | cos[{lo:.2f},{hi:.2f}] -> llm",
                 **q, "llm_calls": calls, "cost_usd": block_cost + calls * c_llm + len(prof.sample) * label_price,
                 "estimated": True, "note": f"LLM part from {n_scored} scored pairs; {esc_share:.1%} escalated"})

    # AC-3: LLM calls saved vs LLM-only, and F1 drop ----------------------------------------------
    llm_only = rows[2]
    for r in rows:
        r["dataset"] = dataset
        r["pc_test"] = pc_test
        r["llm_call_reduction_vs_llm_only"] = 1 - r["llm_calls"] / llm_only["llm_calls"] if llm_only["llm_calls"] else None
        r["f1_drop_vs_llm_only"] = llm_only["f1"] - r["f1"]

    secs = time.perf_counter() - t0
    db[dbm.RUN_LOG].insert_one({"kind": "evaluation", "ts": datetime.now(timezone.utc), "dataset": dataset,
                                "objective": res.objective, "req_id": ObjectId(rep.req_id), "rows": rows,
                                "secs": secs, "llm_budget": llm_budget})
    out = PROJECT_ROOT / "experiments" / "results" / f"evaluation_{dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "method", "plan", "precision", "recall", "f1", "llm_calls", "cost_usd", "estimated",
            "pc_test", "llm_call_reduction_vs_llm_only", "f1_drop_vs_llm_only", "pred_cost_usd", "pred_f1", "note"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("evaluation of %s finished in %.0fs -> %s", dataset, secs, out)
    return rows


def prediction_errors(db: Database | None = None, dataset: str | None = None) -> list[dict]:
    """FR-20 / AC-4: predicted vs observed for every optimizer-driven run."""
    db = db if db is not None else dbm.get_db()
    q = {"kind": "execution", "pred": {"$ne": None}}
    if dataset:
        q["dataset"] = dataset
    out = []
    for r in db[dbm.RUN_LOG].find(q, {"explain": 0, "tiers": 0}).sort("ts", -1):
        e = r.get("error") or {}
        out.append({"ts": r["ts"], "dataset": r["dataset"], "plan": r["plan_text"],
                    "pred_cost": r["pred"]["cost_usd"], "obs_cost": r["obs"]["cost_usd"],
                    "pred_f1": r["pred"]["f1"], "obs_f1": r["obs"]["f1_pair"],
                    "cost_rel_error": e.get("cost_rel_error"), "f1_abs_error": e.get("f1_abs_error")})
    return out
