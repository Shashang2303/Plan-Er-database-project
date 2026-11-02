"""PlanER Streamlit demo (T12 prototype; optimizer controls arrive with T15/T20).

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from planer import db as dbm
from planer.config import settings
from planer.datasets import DATASETS
from planer.request import MATCHER_ORDER, Plan

st.set_page_config(page_title="PlanER", page_icon="🧩", layout="wide")
st.title("PlanER: cost-aware entity resolution")
st.caption("Pick a dataset and a physical plan; PlanER runs it inside MongoDB and reports cost and quality. "
           "In Phase 3 the optimizer will choose this plan for you.")

if not dbm.ping():
    st.error(f"Cannot reach MongoDB at {settings.mongo_uri}. Start the stack: "
             "`docker compose -f docker/docker-compose.yml up -d`")
    st.stop()

db = dbm.get_db()
names = set(db.list_collection_names())
loaded = [k for k, s in DATASETS.items() if s.a.name in names and s.b.name in names]
if not loaded:
    st.warning("No dataset is loaded yet. Run `planer ingest abt_buy` (or `all`) first.")
    st.stop()

MATCHER_LABELS = {"rules": "String rules", "fs": "Fellegi-Sunter (EM)", "rf": "Random forest",
                  "llm": f"Local LLM ({settings.llm_models[0]})"}
PATH_LABELS = {"btree": "B-tree key + $lookup", "multikey": "Multikey rare tokens",
               "search": "$search (BM25 top-k)", "vector": "$vectorSearch (HNSW top-k)"}

# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("ER request")
    dataset = st.selectbox("Dataset pair", loaded, format_func=lambda k: f"{k}  ({DATASETS[k].description})")
    obj_type = st.radio("Objective", ["Target F1 θ", "Budget B (USD)"], horizontal=True)
    if obj_type.startswith("Target"):
        objective = {"type": "theta", "value": st.slider("θ", 0.5, 1.0, 0.85, 0.01)}
    else:
        objective = {"type": "budget", "value": st.number_input("Budget (USD)", 0.0, 100.0, 2.0, 0.5)}

    st.header("Physical plan (hand-chosen)")
    path = st.selectbox("Blocking access path", list(PATH_LABELS), index=3, format_func=PATH_LABELS.get)
    k = st.select_slider("Top-k per record", [5, 10, 20], value=10) if path in ("search", "vector") else None
    chosen = st.multiselect("Matcher cascade (runs in cost order)", list(MATCHER_ORDER),
                            default=["rules", "fs", "rf"], format_func=MATCHER_LABELS.get)
    chosen = [m for m in MATCHER_ORDER if m in chosen]
    thresholds = []
    for m in chosen[:-1]:
        lo, hi = st.slider(f"{MATCHER_LABELS[m]}: decide below / above", 0.0, 1.0, (0.3, 0.7), 0.05, key=f"t_{m}")
        thresholds.append((lo, hi))
    if chosen:
        st.caption(f"{MATCHER_LABELS[chosen[-1]]} decides every remaining pair at 0.5.")
    col1, col2 = st.columns(2)
    do_estimate = col1.button("Estimate", use_container_width=True, disabled=not chosen)
    do_run = col2.button("Run plan", type="primary", use_container_width=True, disabled=not chosen)

plan = Plan(path, chosen, thresholds, k=k) if chosen else None

tab_run, tab_block, tab_match, tab_hist, tab_sys = st.tabs(
    ["Run", "Blocking profiles", "Matcher profiles", "Run history", "System status"])

# ------------------------------------------------------------------------------- run tab
with tab_run:
    if do_estimate and plan:
        try:
            from planer import estimate

            st.session_state["estimate"] = estimate(dataset, plan).as_dict()
        except Exception as exc:
            st.session_state.pop("estimate", None)
            st.warning(f"No estimate: {exc}")
    if do_run and plan:
        from planer.executor import run_fixed_plan

        est = st.session_state.get("estimate")
        predicted = {"cost_usd": est["cost_usd"], "f1": est["f1"]} if est and est["plan"] == plan.describe() else None
        with st.spinner("Blocking, matching and clustering inside MongoDB..."):
            try:
                st.session_state["report"] = run_fixed_plan(dataset, plan, objective, predicted).as_dict()
            except Exception as exc:
                st.error(f"Run failed: {exc}")

    est = st.session_state.get("estimate")
    rep = st.session_state.get("report")
    c = st.columns(4)
    c[0].metric("Predicted cost", f"${est['cost_usd']:.4f}" if est else "—")
    c[1].metric("Observed cost", f"${rep['cost_usd']:.4f}" if rep else "—")
    c[2].metric("Predicted F1", f"{est['f1']:.3f}" if est else "—")
    c[3].metric("Observed F1 (test split)", f"{rep['f1_pair']:.3f}" if rep else "—")

    if plan:
        st.subheader("Plan P = (b, M, τ, κ)")
        st.code(plan.describe(), language=None)
    if est:
        with st.expander("Estimate details (Eq. 1)"):
            st.json(est)
    if rep:
        st.subheader("Observed run")
        c = st.columns(6)
        c[0].metric("Candidates", f"{rep['n_candidates']:,}")
        c[1].metric("Pair completeness", f"{rep['pc_test']:.3f}")
        c[2].metric("Precision / recall", f"{rep['precision']:.2f} / {rep['recall']:.2f}")
        c[3].metric("Cluster F1", f"{rep['f1_cluster']:.3f}")
        c[4].metric("LLM calls (cache hits)", f"{rep['llm_calls']} ({rep['cache_hits']})")
        c[5].metric("Time", f"{rep['secs']:.1f}s")
        tiers = pd.DataFrame(rep["tiers"])
        st.markdown("**Where each pair was decided.** Cheap tiers decide the easy pairs; only uncertain ones escalate.")
        long = tiers.melt(id_vars="matcher", value_vars=["n_match", "n_nonmatch", "n_escalated"],
                          var_name="outcome", value_name="pairs")
        st.plotly_chart(px.bar(long, x="matcher", y="pairs", color="outcome", barmode="stack"), use_container_width=True)
        st.dataframe(tiers[["matcher", "lower", "upper", "n_in", "n_match", "n_nonmatch", "n_escalated",
                            "cost_usd", "llm_calls", "cache_hits"]], hide_index=True, use_container_width=True)
        st.caption(f"Cost: blocking ${rep['block_cost_usd']:.5f} + matching ${rep['match_cost_usd']:.5f} "
                   f"+ labels ${rep['label_cost_usd']:.2f}. Transitivity violations: {rep['transitivity_violations']}.")

        from bson import ObjectId

        clusters = list(db[dbm.CLUSTERS].find({"req_id": ObjectId(rep["req_id"])}).sort("size", -1).limit(25))
        if clusters:
            st.markdown(f"**Entity clusters** ({rep['n_clusters']} with ≥ 2 records; largest shown)")
            spec = DATASETS[dataset]
            rows = []
            for cl in clusters:
                titles = []
                for m in cl["members"][:4]:
                    src, rid = m.split("/", 1)
                    attr = spec.a.title_attr if src == spec.a.name else spec.b.title_attr
                    doc = db[src].find_one({"_id": rid}, {attr: 1})
                    titles.append(f"[{src}] {doc.get(attr) if doc else rid}")
                rows.append({"cluster": cl["cluster_id"], "size": cl["size"], "records": " | ".join(titles)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- blocking
with tab_block:
    from planer.blocking.profile import latest_profiles

    profs = latest_profiles(dataset, db)
    if not profs:
        st.info(f"No blocking profile yet. Run `planer profile-blocking {dataset}`.")
    else:
        df = pd.DataFrame(profs)
        df["label"] = df.apply(lambda r: r["path"] + (f" k={int(r['k'])}" if pd.notna(r.get("k")) else ""), axis=1)
        st.markdown("Each access path trades **pair completeness** (share of true matches kept) against the "
                    "number of candidate pairs the matchers must then examine.")
        st.plotly_chart(px.scatter(df, x="n_candidates", y="pair_completeness", text="label", log_x=True,
                                   labels={"n_candidates": "candidate pairs (log)", "pair_completeness": "pair completeness"}),
                        use_container_width=True)
        st.dataframe(df[["path", "k", "backend", "n_candidates", "pair_completeness", "pc_estimation",
                         "reduction_ratio", "secs", "index_size_bytes", "ts"]], hide_index=True, use_container_width=True)
        sel = st.selectbox("explain() output for", df["label"])
        row = df[df["label"] == sel].iloc[0]
        full = db[dbm.RUN_LOG].find_one({"kind": "blocking_profile", "req_id": row["req_id"]}, sort=[("ts", -1)])
        st.json((full or {}).get("explain") or {}, expanded=False)

# ------------------------------------------------------------------------------- matchers
with tab_match:
    doc = db[dbm.RUN_LOG].find_one({"kind": "optimizer_profile", "dataset": dataset}, sort=[("ts", -1)])
    if not doc:
        st.info(f"No matcher profile yet. Run `planer profile-matchers {dataset}`.")
    else:
        s = doc["sample"]
        st.markdown(f"Stratified sample of **{len(s['pairs'])}** labeled pairs "
                    f"({sum(s['labels'])} matches), label cost ${doc['label_cost_usd']:.2f}, built in {doc['secs']:.0f}s.")
        summ = pd.DataFrame(doc["summary"]).T.reset_index(names="matcher")
        st.dataframe(summ, hide_index=True, use_container_width=True)
        st.caption("Precision / recall are weighted by the inverse sampling probability of each stratum. "
                   "Escalation is the share of pairs a tier would pass on at thresholds (0.3, 0.7).")

# ------------------------------------------------------------------------------- history
with tab_hist:
    runs = list(db[dbm.RUN_LOG].find({"kind": "execution", "dataset": dataset}, {"explain": 0}).sort("ts", -1).limit(50))
    if not runs:
        st.info("No executions yet.")
    else:
        hist = pd.DataFrame([{
            "ts": r["ts"], "plan": r["plan_text"], "obs cost": r["obs"]["cost_usd"], "obs F1": r["obs"]["f1_pair"],
            "pred cost": (r.get("pred") or {}).get("cost_usd"), "pred F1": (r.get("pred") or {}).get("f1"),
            "LLM calls": r["obs"]["llm_calls"], "cache hits": r["obs"]["cache_hits"], "secs": r["obs"]["secs"],
        } for r in runs])
        st.dataframe(hist, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- status
with tab_sys:
    if st.button("Check stack"):
        from planer.health import status

        st.json(status())
