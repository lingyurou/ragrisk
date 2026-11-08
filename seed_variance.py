# -*- coding: utf-8 -*-
"""Split-seed variance of the headline numbers (Section 7.2).

Re-runs the main ladder and the ACI controller over independent 50/50
train/test splits of the *same* 756-query BankKB-C set, so that the paper can
report mean +- sd instead of a single split.  The corpus, corruption seed and
query set are held fixed; only the split / stream ordering changes.

Usage:  python3 seed_variance.py [n_seeds]
Writes: seed_variance.json
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_experiments import prepare, fit_scorer, LADDER, FULL
from kbc import policies as P

RHO, EPS, ALPHA, GAMMA = 0.15, 0.05, 0.05, 0.10
SPLIT_SEEDS = [2026, 11, 12, 13, 14]


def split_stream(qs, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(qs))
    tr = [qs[i] for i in idx[: len(qs) // 2]]
    te = sorted([qs[i] for i in idx[len(qs) // 2:]],
                key=lambda q: (q["ctx"]["t_ref"], q["idx"]))
    return tr, te


def aci_run(units, ev, fit):
    lam, recs = 0.5, []
    for q in ev:
        r = P.run(units, [q], FULL, EPS, lam=lam, score_fn=fit)[0]
        recs.append(r)
        lam = min(0.999, max(0.001, lam + GAMMA * (r["viol"] - ALPHA)))
    sm = P.summarize(recs)
    return dict(bcr=sm["bcr"], harm=sm["overall"], coverage=sm["coverage"])


def main():
    seeds = SPLIT_SEEDS[: int(sys.argv[1])] if len(sys.argv) > 1 else SPLIT_SEEDS
    units, qs = prepare(RHO)
    rows = []
    for s in seeds:
        tr, te = split_stream(qs, s)
        fit = fit_scorer(units, tr)
        row = {"seed": s, "n_test": len(te)}
        for name in ("M1_vanilla", "M4_asof", "M6_resolve"):
            sm = P.summarize(P.run(units, te, LADDER[name], EPS))
            row[name] = dict(bcr=sm["bcr"], harm=sm["overall"], coverage=sm["coverage"])
        # the controller is run on the second half of the stream, as in evaluate()
        row["M7_aci"] = aci_run(units, te[len(te) // 2:], fit)
        rows.append(row)
        print("seed", s, {k: round(v["bcr"], 3) if isinstance(v, dict) else v
                          for k, v in row.items() if isinstance(v, dict) or k == "seed"},
              flush=True)

    summary = {}
    for name in ("M1_vanilla", "M4_asof", "M6_resolve", "M7_aci"):
        for metric in ("bcr", "harm", "coverage"):
            vals = np.array([r[name][metric] for r in rows], dtype=float)
            summary.setdefault(name, {})[metric] = dict(
                mean=float(vals.mean()), sd=float(vals.std(ddof=1)), n=int(vals.size))

    json.dump({"config": dict(rho=RHO, eps=EPS, alpha=ALPHA, gamma=GAMMA),
               "seeds": seeds, "runs": rows, "summary": summary},
              open(os.path.join(HERE, "seed_variance.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
