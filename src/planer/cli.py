"""Command line interface: `planer <command>` (or `python -m planer <command>`).

Typical first run (see README):
    planer status
    planer ingest all                  # download + normalize + embed + load (T2)
    planer index all                   # $search / $vectorSearch indexes on mongot (T5, T6)
    planer llm-pull                    # download the local LLM into Ollama (T11)
    planer profile-blocking abt_buy    # T7
    planer profile-matchers abt_buy    # T13 (+ stratified sample, T14)
    planer estimate abt_buy --path vector --k 10 --matchers rules,fs,rf,llm
    planer run abt_buy --path vector --k 10 --matchers rules,fs,rf,llm --estimate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .datasets import DATASETS


def _datasets(arg: str) -> list[str]:
    return list(DATASETS) if arg == "all" else [arg]


def _thresholds(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        lo, hi = part.split(":")
        out.append((float(lo), float(hi)))
    return out


def _plan(args):
    from .request import Plan

    return Plan(
        access_path=args.path, k=args.k, matchers=[m.strip() for m in args.matchers.split(",")],
        thresholds=_thresholds(args.thresholds),
    )


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_status(args) -> None:
    from .health import status

    _print(status())


def cmd_download(args) -> None:
    from .datasets import get_dataset
    from .ingest import download

    for key in _datasets(args.dataset):
        print(f"{key}: {download(get_dataset(key), force=args.force)}")


def cmd_ingest(args) -> None:
    from .ingest import ingest_dataset

    for key in _datasets(args.dataset):
        _print(ingest_dataset(key))


def cmd_bkeys(args) -> None:
    from .ingest import compare_bkey_rules, rekey

    for key in _datasets(args.dataset):
        print(f"\n{key}")
        print(f"{'rule':<15}{'candidates':>12}{'PC':>8}{'RR':>9}")
        for r in compare_bkey_rules(key):
            mark = "  <- current" if r["current"] else ""
            print(f"{r['rule']:<15}{r['candidates']:>12}{r['pair_completeness']:>8.3f}{r['reduction_ratio']:>9.4f}{mark}")
        if args.apply:
            _print(rekey(key, args.apply if args.apply != "current" else None))


def cmd_index(args) -> None:
    from .datasets import get_dataset
    from .search_indexes import create_search_indexes

    for key in _datasets(args.dataset):
        for src in get_dataset(key).sources:
            _print({src: create_search_indexes(src, wait=not args.no_wait)})


def cmd_llm_pull(args) -> None:
    from .config import settings
    from .matchers.llm import ollama_pull

    for model in args.models or settings.llm_models:
        ollama_pull(model)
        print(f"pulled {model}")


def cmd_profile_blocking(args) -> None:
    from .blocking.profile import profile_all

    ks = tuple(int(k) for k in args.ks.split(","))
    paths = tuple(args.paths.split(",")) if args.paths else None
    for key in _datasets(args.dataset):
        rows = profile_all(key, ks=ks, paths=paths)
        print(f"\n{key}")
        print(f"{'path':<9}{'k':>5}{'candidates':>12}{'PC':>8}{'RR':>9}{'secs':>8}  indexes used")
        for r in rows:
            if "error" in r:
                print(f"{r['path']:<9}{str(r['k']):>5}  ERROR {r['error']}")
                continue
            used = ",".join((r.get("explain") or {}).get("summary", {}).get("indexes_used", [])) or "-"
            print(f"{r['path']:<9}{str(r['k'] or '-'):>5}{r['n_candidates']:>12}{r['pair_completeness']:>8.3f}"
                  f"{r['reduction_ratio']:>9.4f}{r['secs']:>8.1f}  {used}")


def cmd_profile_matchers(args) -> None:
    from .optimizer.profiling import profile_matchers, summarize

    matchers = tuple(m.strip() for m in args.matchers.split(","))
    prof = profile_matchers(args.dataset, args.sample_size, matchers, args.llm_sample)
    print(f"sample of {len(prof.sample)} pairs ({int(prof.sample.labels.sum())} matches), built in {prof.secs:.1f}s")
    _print(summarize(prof))


def cmd_estimate(args) -> None:
    from . import estimate

    _print(estimate(args.dataset, _plan(args)).as_dict())


def cmd_run(args) -> None:
    from .executor import run_fixed_plan

    plan = _plan(args)
    predicted = None
    if args.estimate:
        from . import estimate

        est = estimate(args.dataset, plan)
        predicted = {"cost_usd": est.cost_usd, "f1": est.f1, "precision": est.precision, "recall": est.recall}
        print(f"predicted: cost=${est.cost_usd:.4f}  F1={est.f1:.3f}")
    objective = {"type": "theta", "value": args.theta} if args.budget is None else {"type": "budget", "value": args.budget}
    report = run_fixed_plan(args.dataset, plan, objective=objective, predicted=predicted)
    print(f"plan: {plan.describe()}")
    _print(report.as_dict())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="planer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    ds_help = f"one of {', '.join(DATASETS)} or 'all'"

    sub.add_parser("status", help="check MongoDB, mongot, Ollama and loaded data").set_defaults(func=cmd_status)

    p = sub.add_parser("download", help="download benchmark archives")
    p.add_argument("dataset", help=ds_help)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("ingest", help="download, normalize, embed and load a benchmark (T2)")
    p.add_argument("dataset", help=ds_help)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("bkeys", help="compare blocking-key rules; --apply recomputes bkey in place")
    p.add_argument("dataset", help=ds_help)
    p.add_argument("--apply", nargs="?", const="current", default=None,
                   help="rule to apply (default: the dataset's configured rule)")
    p.set_defaults(func=cmd_bkeys)

    p = sub.add_parser("index", help="create $search and $vectorSearch indexes (T5, T6)")
    p.add_argument("dataset", help=ds_help)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("llm-pull", help="download local LLMs into Ollama")
    p.add_argument("models", nargs="*")
    p.set_defaults(func=cmd_llm_pull)

    p = sub.add_parser("profile-blocking", help="profile all access paths (T7)")
    p.add_argument("dataset", help=ds_help)
    p.add_argument("--ks", default="5,10,20", help="k values for the top-k paths")
    p.add_argument("--paths", default=None, help="subset, e.g. btree,multikey")
    p.set_defaults(func=cmd_profile_blocking)

    p = sub.add_parser("profile-matchers", help="stratified sample + per-matcher profiling (T13, T14)")
    p.add_argument("dataset")
    p.add_argument("--sample-size", type=int, default=300)
    p.add_argument("--llm-sample", type=int, default=60)
    p.add_argument("--matchers", default="rules,fs,rf,llm")
    p.set_defaults(func=cmd_profile_matchers)

    for name, func, helptext in (
        ("estimate", cmd_estimate, "predict the cost and F1 of a plan (T14)"),
        ("run", cmd_run, "execute a fixed plan end to end (T12)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("dataset")
        p.add_argument("--path", default="vector", choices=["btree", "multikey", "search", "vector"])
        p.add_argument("--k", type=int, default=None)
        p.add_argument("--matchers", default="rules,fs,rf")
        p.add_argument("--thresholds", default=None, help="l:u per non-final tier, e.g. 0.2:0.8,0.3:0.7")
        if name == "run":
            p.add_argument("--estimate", action="store_true", help="also log the predicted cost and F1")
            p.add_argument("--theta", type=float, default=0.85)
            p.add_argument("--budget", type=float, default=None)
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "sentence_transformers", "pymongo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
