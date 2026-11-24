"""PlanER Streamlit demo (FR-21, T20).

Optimizer mode: state a target F1 or a budget, PlanER enumerates and prunes plans once, then every
move of the objective re-selects from the cached Pareto set (no re-estimation, AC-8).
Hand-chosen mode: the Phase-2 prototype, for comparing any plan against the optimizer's choice.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from bson import ObjectId

from planer import db as dbm
from planer.config import settings
from planer.datasets import DATASETS
from planer.request import MATCHER_ORDER, Plan

st.set_page_config(page_title="PlanER", page_icon="🧩", layout="wide")
st.title("PlanER: cost-aware entity resolution")
st.caption("State what quality you need or what you can spend. PlanER chooses how to find candidate pairs, "
           "which matchers to run and when to ask the LLM, runs the plan inside MongoDB and reports what it cost.")

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


def plain_plan(plan: Plan) -> str:
    """The plan in words, for readers without MongoDB or ML background (NFR-8)."""
    path = PATH_LABELS[plan.access_path] + (f", {plan.k} neighbours per record" if plan.k else "")
    steps = []
    for i, (m, (lo, hi)) in enumerate(zip(plan.matchers, plan.thresholds)):
        if i < len(plan.matchers) - 1:
            steps.append(f"{MATCHER_LABELS[m.split(':')[0]]} accepts pairs scoring above {hi:.2f}, rejects below "
                         f"{lo:.2f} and passes the rest on")
        else:
            steps.append(f"{MATCHER_LABELS[m.split(':')[0]]} decides every remaining pair")
    repair = "then splits clusters that contradict a confident non-match" if plan.clustering == "cc_repair" else ""
    return f"Find candidates with **{path}**. " + "; ".join(steps) + ". Group matches into entities" + (
        f", {repair}." if repair else ".")


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("ER request")
    dataset = st.selectbox("Dataset pair", loaded, format_func=lambda k: f"{k}  ({DATASETS[k].description})")
    obj_type = st.radio("Objective", ["Target F1 θ", "Budget B (USD)"], horizontal=True)
    if obj_type.startswith("Target"):
        objective = {"type": "theta", "value": st.slider("θ (minimum F1)", 0.3, 1.0, 0.85, 0.01)}
    else:
        objective = {"type": "budget", "value": st.slider("Budget (USD)", 0.0, 100.0, 10.0, 0.5)}

    mode = st.radio("Who chooses the plan?", ["Optimizer", "Me (hand-chosen)"], horizontal=True)
    if mode == "Optimizer":
        do_optimize = st.button("Optimize", width="stretch",
                                help="Enumerate and cost every plan on the labeled sample (done once per dataset)")
        do_run = st.button("Run chosen plan", type="primary", width="stretch",
                           disabled="opt_req" not in st.session_state or st.session_state.get("opt_ds") != dataset)
        do_estimate = False
    else:
        do_optimize = False
        st.subheader("Physical plan")
        path = st.selectbox("Blocking access path", list(PATH_LABELS), index=3, format_func=PATH_LABELS.get)
        k = st.select_slider("Top-k per record", [5, 10, 20], value=10) if path in ("search", "vector") else None
        chosen = st.multiselect("Matcher cascade (runs in cost order)", list(MATCHER_ORDER),
                                default=["rules", "rf"], format_func=MATCHER_LABELS.get)
        chosen = [m for m in MATCHER_ORDER if m in chosen]
        thresholds = []
        for m in chosen[:-1]:
            lo, hi = st.slider(f"{MATCHER_LABELS[m]}: reject below / accept above", 0.0, 1.0, (0.2, 0.8), 0.05,
                               key=f"t_{m}")
            thresholds.append((lo, hi))
        repair = st.checkbox("Transitivity repair", value=True)
        col1, col2 = st.columns(2)
        do_estimate = col1.button("Estimate", width="stretch", disabled=not chosen)
        do_run = col2.button("Run plan", type="primary", width="stretch", disabled=not chosen)

# ------------------------------------------------------------------------------- plan selection
plan, predicted, feasible, front = None, None, True, []
if mode == "Optimizer":
    from planer.optimizer import optimize, reselect, stored_pareto
    from planer.request import default_request, submit

    if do_optimize:
        with st.spinner("Enumerating and costing plans on the labeled sample..."):
            try:
                req_id = submit(default_request(dataset, objective), db)
                res = optimize(req_id, db)
                st.session_state.update(opt_req=str(req_id), opt_ds=dataset, opt_secs=res.secs,
                                        opt_n=res.n_plans)
                # the sidebar's "Run chosen plan" button was drawn before this result existed, so it is
                # still disabled; rerun once so it is redrawn as enabled
                st.rerun()
            except Exception as exc:
                st.error(f"Optimizer failed: {exc}. Profile the dataset first: "
                         f"`planer profile-blocking {dataset}` and `planer profile-matchers {dataset}`.")
    if st.session_state.get("opt_ds") == dataset and "opt_req" in st.session_state:
        t0 = time.perf_counter()
        point, feasible = reselect(st.session_state["opt_req"], objective, db)   # cached Pareto set: fast
        front = stored_pareto(st.session_state["opt_req"], db)
        st.session_state["reselect_ms"] = (time.perf_counter() - t0) * 1000
        plan = point.plan
        predicted = {"cost_usd": point.cost, "f1": point.f1, "precision": point.est.precision,
                     "recall": point.est.recall}
else:
    if chosen:
        plan = Plan(path, chosen, thresholds, k=k, clustering="cc_repair" if repair else "connected_components")
    if do_estimate and plan:
        try:
            from planer import estimate

            e = estimate(dataset, plan)
            st.session_state["estimate"] = {"plan": plan.describe(), "cost_usd": e.cost_usd, "f1": e.f1,
                                            "precision": e.precision, "recall": e.recall}
        except Exception as exc:
            st.session_state.pop("estimate", None)
            st.warning(f"No estimate: {exc}")
    est = st.session_state.get("estimate")
    if est and plan and est["plan"] == plan.describe():
        predicted = {k_: est[k_] for k_ in ("cost_usd", "f1", "precision", "recall")}

tab_run, tab_eval, tab_block, tab_match, tab_hist, tab_sys = st.tabs(
    ["Plan & run", "Evaluation", "Blocking profiles", "Matcher profiles", "Run history", "System status"])

# ------------------------------------------------------------------------------- run tab
with tab_run:
    if do_run and plan:
        from planer.executor import execute
        from planer.request import default_request, submit

        sample_cost = 0.0
        if mode == "Optimizer":
            doc = db[dbm.REQUESTS].find_one({"_id": ObjectId(st.session_state["opt_req"])}, {"optimizer": 1})
            sample_cost = doc["optimizer"]["sample_label_cost_usd"]
        with st.spinner("Blocking, matching and clustering inside MongoDB..."):
            try:
                req_id = submit(default_request(dataset, objective), db)
                st.session_state["report"] = execute(req_id, plan, predicted=predicted,
                                                     sample_label_cost=sample_cost, db=db).as_dict()
                st.session_state["report_plan"] = plan.describe()
            except Exception as exc:
                st.error(f"Run failed: {exc}")

    rep = st.session_state.get("report")
    if rep and plan and st.session_state.get("report_plan") != plan.describe():
        rep = None  # the shown plan changed since the last run
    c = st.columns(4)
    c[0].metric("Predicted cost", f"${predicted['cost_usd']:.2f}" if predicted else "—")
    c[1].metric("Observed cost", f"${rep['cost_usd']:.2f}" if rep else "—")
    c[2].metric("Predicted F1", f"{predicted['f1']:.3f}" if predicted else "—")
    c[3].metric("Observed F1 (test split)", f"{rep['f1_pair']:.3f}" if rep else "—",
                delta=f"{rep['f1_pair'] - predicted['f1']:+.3f} vs predicted" if rep and predicted else None)

    if mode == "Optimizer" and not plan:
        st.info("Press **Optimize** to let PlanER choose a plan for this dataset.")
    if plan:
        st.subheader("Chosen plan P = (b, M, τ, κ)" if mode == "Optimizer" else "Your plan P = (b, M, τ, κ)")
        if mode == "Optimizer" and not feasible:
            st.warning("No plan meets this objective; showing the closest one "
                       "(the most accurate plan for a target F1, the cheapest for a budget).")
        st.markdown(plain_plan(plan))
        st.code(plan.describe(), language=None)
        if mode == "Optimizer":
            st.caption(f"{st.session_state.get('opt_n', 0):,} plans costed in {st.session_state.get('opt_secs', 0):.1f}s; "
                       f"re-selecting for a new objective took {st.session_state.get('reselect_ms', 0):.0f} ms.")

    if front:
        df = pd.DataFrame([{"cost": p.cost, "F1": p.f1, "plan": p.plan.describe()} for p in front])
        fig = px.line(df, x="cost", y="F1", markers=True, hover_data=["plan"],
                      labels={"cost": "estimated cost (USD)", "F1": "estimated F1"})
        fig.add_trace(go.Scatter(x=[predicted["cost_usd"]], y=[predicted["f1"]], mode="markers",
                                 marker=dict(size=16, symbol="star"), name="chosen plan"))
        if objective["type"] == "theta":
            fig.add_hline(y=objective["value"], line_dash="dot", annotation_text="target F1")
        else:
            fig.add_vline(x=objective["value"], line_dash="dot", annotation_text="budget")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          title="Pareto frontier of candidate plans")
        st.plotly_chart(fig, width="stretch")

    if rep:
        st.subheader("Observed run")
        c = st.columns(6)
        c[0].metric("Candidates", f"{rep['n_candidates']:,}")
        c[1].metric("Pair completeness", f"{rep['pc_test']:.3f}")
        c[2].metric("Precision / recall", f"{rep['precision']:.2f} / {rep['recall']:.2f}")
        c[3].metric("Cluster F1", f"{rep['f1_cluster']:.3f}")
        c[4].metric("LLM calls (cache hits)", f"{rep['llm_calls']} ({rep['cache_hits']})")
        c[5].metric("Time", f"{rep['secs']:.1f}s")
        if rep.get("error"):
            e = rep["error"]
            st.caption(f"Prediction error: cost {e['cost_rel_error']:.0%}, F1 {e['f1_abs_error']:.3f} "
                       f"(targets: 15% and 0.05).")
        if rep.get("repair"):
            r = rep["repair"]
            st.caption(f"Transitivity repair: {r['violations_before']} → {r['violations_after']} violations, "
                       f"{r['edges_removed']} weak links removed, no LLM calls.")
        if rep.get("budget"):
            b = rep["budget"]
            (st.error if b["over_budget"] else st.success)(
                f"Budget ${b['budget_usd']:.2f}: {'exceeded, ' + b['cause'] if b['over_budget'] else 'respected'}.")
        tiers = pd.DataFrame(rep["tiers"])
        st.markdown("**Where each pair was decided.** Cheap tiers decide the easy pairs; only uncertain ones escalate.")
        long = tiers.melt(id_vars="matcher", value_vars=["n_match", "n_nonmatch", "n_escalated"],
                          var_name="outcome", value_name="pairs")
        st.plotly_chart(px.bar(long, x="matcher", y="pairs", color="outcome", barmode="stack"), width="stretch")
        st.caption(f"Cost: blocking ${rep['block_cost_usd']:.5f} + matching ${rep['match_cost_usd']:.5f} "
                   f"+ labels ${rep['label_cost_usd']:.2f}.")

        clusters = list(db[dbm.CLUSTERS].find({"req_id": ObjectId(rep["req_id"])}).sort("size", -1).limit(20))
        if clusters:
            st.markdown(f"**Entity clusters** ({rep['n_clusters']} with ≥ 2 records; largest shown)")
            spec = DATASETS[dataset]
            rows = []
            for cl in clusters:
                titles = []
                for m in cl["members"][:4]:
                    src, rid = m.split("/", 1)
                    attr = spec.a.title_attr if src == spec.a.name else spec.b.title_attr
                    d = db[src].find_one({"_id": rid}, {attr: 1})
                    titles.append(f"[{src}] {d.get(attr) if d else rid}")
                rows.append({"cluster": cl["cluster_id"], "size": cl["size"], "records": " | ".join(titles)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------- evaluation
with tab_eval:
    ev = db[dbm.RUN_LOG].find_one({"kind": "evaluation", "dataset": dataset}, sort=[("ts", -1)])
    if not ev:
        st.info(f"No evaluation yet. Run `planer evaluate {dataset}`.")
    else:
        st.markdown(f"PlanER against three baselines on the same candidates and held-out test split "
                    f"(objective {ev['objective']['type']} = {ev['objective']['value']}). "
                    "Rows marked *estimated* score a random sample of pairs with the LLM instead of all of them.")
        df = pd.DataFrame(ev["rows"])
        st.dataframe(df[["method", "precision", "recall", "f1", "llm_calls", "cost_usd", "estimated",
                         "llm_call_reduction_vs_llm_only", "f1_drop_vs_llm_only"]],
                     hide_index=True, width="stretch")
        st.plotly_chart(px.scatter(df, x="cost_usd", y="f1", text="method", log_x=True,
                                   labels={"cost_usd": "cost (USD, log)", "f1": "pairwise F1"}),
                        width="stretch")

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
                        width="stretch")
        st.dataframe(df[["path", "k", "backend", "n_candidates", "pair_completeness", "pc_estimation",
                         "reduction_ratio", "secs", "index_size_bytes", "ts"]], hide_index=True, width="stretch")
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
        st.dataframe(summ, hide_index=True, width="stretch")
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
            "cost err": (r.get("error") or {}).get("cost_rel_error"), "F1 err": (r.get("error") or {}).get("f1_abs_error"),
            "LLM calls": r["obs"]["llm_calls"], "cache hits": r["obs"]["cache_hits"], "secs": r["obs"]["secs"],
        } for r in runs])
        st.dataframe(hist, hide_index=True, width="stretch")

# ------------------------------------------------------------------------------- status
with tab_sys:
    if st.button("Check stack"):
        from planer.health import status

        st.json(status())
