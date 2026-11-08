# -*- coding: utf-8 -*-
"""Driver for the BIND-RAG experiments on BankKB-C."""
import json, os, sys, zlib, dataclasses
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kbc.bench import build_kb, build_queries, FAMILIES
from kbc import policies as P

OUT = os.path.join(HERE, "results.json")
K = 50

LADDER = {
    "M1_vanilla": dict(),
    "M2_slot": dict(slot=True),
    "M3_recency": dict(slot=True, temporal="now"),
    "M4_asof": dict(slot=True, temporal="ref"),
    "M5_scope": dict(slot=True, temporal="ref", scope=True, authority=True),
    "M6_resolve": dict(slot=True, temporal="ref", scope=True, resolve=True),
}
ABL = {
    "A_no_temporal": dict(slot=True, scope=True, resolve=True),
    "A_no_scope": dict(slot=True, temporal="ref", resolve=True),
    "A_no_slot": dict(temporal="ref", scope=True, resolve=True),
    "A_no_grant": dict(slot=True, temporal="ref", scope=True, resolve=True, grant_rule=False),
}
FULL = LADDER["M6_resolve"]


def corrupt(units, rho, seed=11):
    """Field-level metadata corruption of the *system view*; gold stays clean."""
    rng = np.random.default_rng(seed)
    out = []
    for u in units:
        v = dataclasses.replace(u)
        if rng.random() < rho:
            kind = rng.integers(0, 5)
            if kind == 0 and v.valid_to < 999:
                v.valid_to = 999                       # expired clause looks current
            elif kind == 1:
                v.valid_from = min(30, v.valid_from + 12)   # delayed enactment
            elif kind == 2 and v.tier in ("L2", "L3"):
                v.tier = "L4" if v.tier == "L2" else "L5"   # mislabelled authority
            elif kind == 3 and v.region != "全部":
                v.region = "全部"                        # over-generalised scope
            else:
                try:
                    v.value = round(float(v.value) / 10.0, 2)  # OCR digit loss
                except (TypeError, ValueError):
                    pass
        out.append(v)
    return out


def prepare(rho=0.0):
    units_clean, T = build_kb()
    units = corrupt(units_clean, rho) if rho > 0 else units_clean
    qs = build_queries(units_clean, T)
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    for q in qs:
        f = P.rrf(bm.scores(q["question"]), dn.scores(q["question"]))
        q["_top"], q["_fs"] = list(np.argsort(-f)[:K]), f
    return units, qs


def fit_scorer(units, train_qs, eps=0.05, seed=7):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for q in train_qs:
        best, f, ctx = P.assemble(units, q, q["_top"], FULL)
        if best is None:
            X.append(P.vec(f)); y.append(0); continue
        v = P.pick(ctx, q, [units[i] for i in q['_top']], eps, rng)
        X.append(P.vec(f)); y.append(int(v == q["gold_value"]))
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    return lambda f: float(lr.predict_proba(sc.transform([P.vec(f)]))[0, 1])


def curve(recs):
    errs, cov, risk = 0, [], []
    for i, r in enumerate(recs, 1):
        errs += r["viol"]
        risk.append(errs / i)
        cov.append(sum(x["answered"] for x in recs[:i]) / i)
    return risk, cov


def evaluate(units, test_qs, fit, eps, alpha, beta=3.0, gamma=0.10):
    out = {}
    for name, cfg in LADDER.items():
        recs = P.run(units, test_qs, cfg, eps, beta=beta)
        s = P.summarize(recs)
        s["by_family"] = {f: P.summarize([r for r in recs if r["family"] == f]) for f in FAMILIES}
        out[name] = s

    for name, cfg in ABL.items():
        recs = P.run(units, test_qs, cfg, eps, lam=0.0, score_fn=fit, beta=beta)
        s = P.summarize(recs)
        s["by_family"] = {f: P.summarize([r for r in recs if r["family"] == f]) for f in FAMILIES}
        out[name] = s

    n = len(test_qs)
    cal, ev = test_qs[: n // 2], test_qs[n // 2:]

    best_lam, best_cov = 0.995, -1
    for lam in np.linspace(0.02, 0.99, 80):
        s = P.summarize(P.run(units, cal, FULL, eps, lam=float(lam), score_fn=fit, beta=beta))
        if s["bcr"] <= alpha and s["coverage"] > best_cov:
            best_lam, best_cov = float(lam), s["coverage"]
    recs = P.run(units, ev, FULL, eps, lam=best_lam, score_fn=fit, beta=beta)
    s = P.summarize(recs); s["lambda"] = best_lam
    s["by_family"] = {f: P.summarize([r for r in recs if r["family"] == f]) for f in FAMILIES}
    out["M6b_static"] = s

    lam, recs = 0.5, []
    for q in ev:
        r = P.run(units, [q], FULL, eps, lam=lam, score_fn=fit, beta=beta)[0]
        recs.append(r)
        lam = min(0.999, max(0.001, lam + gamma * (r["viol"] - alpha)))
    s = P.summarize(recs)
    s["by_family"] = {f: P.summarize([r for r in recs if r["family"] == f]) for f in FAMILIES}
    s["risk_curve"], s["cov_curve"] = curve(recs)
    s["family_stream"] = [r["family"] for r in recs]
    out["M7_aci"] = s

    recs0 = P.run(units, ev, FULL, eps, lam=0.0, score_fn=fit, beta=beta)
    s0 = P.summarize(recs0)
    s0["by_family"] = {f: P.summarize([r for r in recs0 if r["family"] == f]) for f in FAMILIES}
    out["A_no_control"] = s0
    return out


def risk_coverage(units, test_qs, fit, eps):
    rows = {}
    for name in ["M1_vanilla", "M4_asof", "M5_scope", "M6_resolve"]:
        cfg = LADDER[name]
        recs = []
        for q in test_qs:
            best, f, ctx = P.assemble(units, q, q["_top"], cfg)
            # stable across processes: Python's str hash is randomised per run
            rng = np.random.default_rng(zlib.crc32(str(q["qid"]).encode()) % 2**31)
            val = P.pick(ctx, q, [units[i] for i in q["_top"]], eps, rng) if best is not None else None
            recs.append(dict(viol=int(best is not None and val != q["gold_value"]),
                             s=fit(f) if name == "M6_resolve" else float(f["score"])))
        order = sorted(recs, key=lambda r: -r["s"])
        risk, cov, e = [], [], 0
        for i, r in enumerate(order, 1):
            e += r["viol"]; risk.append(e / i); cov.append(i / len(order))
        rows[name] = dict(risk=risk, coverage=cov)
    return rows


def main():
    RHO = 0.15
    units, qs = prepare(RHO)
    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(qs))
    tr = [qs[i] for i in idx[: len(qs) // 2]]
    te = sorted([qs[i] for i in idx[len(qs) // 2:]], key=lambda q: (q["ctx"]["t_ref"], q["idx"]))
    fit = fit_scorer(units, tr)

    res = {"meta": dict(units=len(units), queries=len(qs), train=len(tr), test=len(te),
                        k=K, rho=RHO, seed=20261029,
                        families={f: sum(1 for q in qs if q["family"] == f) for f in FAMILIES}),
           "grid": {}, "alpha": {}, "rho": {}}

    for eps in [0.0, 0.05, 0.10]:
        res["grid"][f"e{eps}"] = evaluate(units, te, fit, eps, 0.05)
        print("eps", eps, flush=True)
    for alpha in [0.10, 0.02]:
        res["alpha"][f"a{alpha}"] = evaluate(units, te, fit, 0.05, alpha)
        print("alpha", alpha, flush=True)

    # robustness to metadata corruption
    for rho in [0.0, 0.10, 0.25]:
        u2, qs2 = prepare(rho)
        r2 = np.random.default_rng(2026)
        i2 = r2.permutation(len(qs2))
        tr2 = [qs2[i] for i in i2[: len(qs2) // 2]]
        te2 = sorted([qs2[i] for i in i2[len(qs2) // 2:]], key=lambda q: (q["ctx"]["t_ref"], q["idx"]))
        f2 = fit_scorer(u2, tr2)
        res["rho"][f"r{rho}"] = evaluate(u2, te2, f2, 0.05, 0.05)
        print("rho", rho, flush=True)

    res["risk_coverage"] = risk_coverage(units, te, fit, 0.05)
    json.dump(res, open(OUT, "w"), ensure_ascii=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
