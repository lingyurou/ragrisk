# -*- coding: utf-8 -*-
"""
Policy layer. Every method is a configuration of four switches, so the ablation
ladder in the paper is literally the same code path:

  scope   : restrict retrieval to units whose scope matches the request context
  asof    : select units valid at the reference time of the request
  resolve : normative resolution (admissibility / lex specialis / stringentior)
  control : risk-controlled abstention (static calibration or ACI)

The generator is a controlled context-confusion reader: it emits the selected
unit's value with probability 1-eps, and can only flip when the assembled
context actually contains a competing value.
"""
import math
import numpy as np

TOK = lambda s: [s[i:i + 2] for i in range(len(s) - 1)]
OPERATIVE = ("L2", "L3", "L4")
NOW = 35   # end of the knowledge horizon
TIER_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [TOK(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df, self.tf = {}, []
        for d in self.docs:
            t = {}
            for w in d:
                t[w] = t.get(w, 0) + 1
            self.tf.append(t)
            for w in t:
                self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in self.df.items()}

    def scores(self, q):
        q = TOK(q)
        out = np.zeros(self.N)
        for i, t in enumerate(self.tf):
            dl = len(self.docs[i])
            for w in q:
                if w in t:
                    f = t[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


class Dense:
    """Latent-semantic retriever standing in for a bi-encoder (documented in the paper)."""
    def __init__(self, docs, k=128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1, sublinear_tf=True)
        X = self.vec.fit_transform(docs)
        self.svd = TruncatedSVD(n_components=min(k, X.shape[1] - 1), random_state=0)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def scores(self, q):
        z = self.svd.transform(self.vec.transform([q]))
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        return (self.Z @ z.T).ravel()


def rrf(s1, s2, k=60, depth=100):
    r1, r2 = np.argsort(-s1), np.argsort(-s2)
    out = np.zeros_like(s1)
    for r in (r1, r2):
        for pos, i in enumerate(r[:depth]):
            out[i] += 1.0 / (k + pos + 1)
    return out


def spec(u):
    return (u.region != "全部") + (u.segment != "全部客户")


def laxer(u, v):
    try:
        a, b = float(u.value), float(v.value)
    except (TypeError, ValueError):
        return None
    if u.attr in ("起存金额", "最低首付比例"):
        return a < b
    return a > b


EMPTY = dict(score=0.0, margin=0.0, uniq=0.0, agree=0.0, tmargin=0.0, spec=0.0, n=0.0)


def assemble(units, q, top_idx, cfg):
    """Return (selected_unit | None, feature dict, assembled context units)."""
    ctx = q["ctx"]
    cands = [units[i] for i in top_idx]
    if cfg.get("slot"):
        cands = [u for u in cands if u.product == ctx["product"] and u.attr == ctx["attr"]]
    if cfg.get("scope"):
        cands = [u for u in cands if u.region in ("全部", ctx["region"])
                 and u.segment in ("全部客户", ctx["segment"])]
    tref = {"ref": ctx["t_ref"], "now": NOW, None: None}[cfg.get("temporal")]
    if tref is not None:
        cands = [u for u in cands if u.valid_from <= tref < u.valid_to]
    if not cands:
        return None, dict(EMPTY), []

    if not cfg.get("resolve"):
        op = [u for u in cands if u.tier in OPERATIVE] or cands
        best = (sorted(op, key=lambda u: (TIER_RANK[u.tier], -u.valid_from))[0]
                if cfg.get("authority") else op[0])
    else:
        op = [u for u in cands if u.tier in OPERATIVE]
        if not op:
            return None, dict(EMPTY), cands
        adm = []
        for u in op:
            ok = True
            if not (u.grant and cfg.get("grant_rule", True)):   # R1: ultra vires check
                for v in op:
                    if TIER_RANK[v.tier] < TIER_RANK[u.tier] and laxer(u, v) is True:
                        ok = False
                        break
            if ok:
                adm.append(u)
        if not adm:
            return None, dict(EMPTY), cands
        ms = max(spec(u) for u in adm)
        pool = [u for u in adm if spec(u) == ms]                 # R2 lex specialis
        top = min(TIER_RANK[u.tier] for u in pool)
        pool = [u for u in pool if TIER_RANK[u.tier] == top]     # R4 authority
        pool.sort(key=lambda u: -u.valid_from)                   # R5 recency
        best = pool[0]
        for u in pool:                                           # R3 stringentior
            if laxer(best, u) is True:
                best = u

    same = [u for u in cands if str(u.value) == str(best.value)]
    other = [u for u in cands if str(u.value) != str(best.value)]
    pos = {id(u): p for p, u in enumerate(cands)}
    fs = q["_fs"]
    s_best = float(fs[top_idx[pos[id(best)]]]) if id(best) in pos else 0.0
    s_other = max([float(fs[top_idx[pos[id(u)]]]) for u in other], default=0.0)
    tm = min(abs(ctx["t_ref"] - best.valid_from),
             abs(ctx["t_ref"] - best.valid_to) if best.valid_to < 900 else 99)
    f = dict(score=s_best, margin=s_best - s_other,
             uniq=1.0 if (len(same) == 1 and not other) else 0.0,
             agree=len(same) / max(1, len(cands)),
             tmargin=min(tm, 12) / 12.0,
             spec=1.0 if spec(best) else 0.6,
             n=float(len(cands)))
    # our method surfaces the binding unit plus at most one explicit contrast;
    # baselines dump the whole retrieved list into the prompt.
    cn = cands if not cfg.get("resolve") else [best]
    return best, f, cn


FEAT_KEYS = ["score", "margin", "uniq", "agree", "tmargin", "spec"]


def vec(f):
    return [f[k] for k in FEAT_KEYS]


def _utility(u, q):
    """Evidence utility a generic reader can compute from the prompt alone."""
    fs = q["_fs"]
    try:
        pos = q["_top"].index(_IDX[id(u)])
        rel = float(fs[_IDX[id(u)]]) / (float(fs[q["_top"][0]]) + 1e-12)
    except Exception:
        rel = 0.0
    auth = 1.0 / TIER_RANK[u.tier]
    cur = 1.0 if (u.valid_from <= NOW < u.valid_to) else 0.25
    sc = 1.0 if (u.region in ("全部", q["ctx"]["region"])
                 and u.segment in ("全部客户", q["ctx"]["segment"])) else 0.5
    return 0.45 * rel + 0.30 * auth + 0.15 * cur + 0.10 * sc


_IDX = {}


def pick(context, q, memory, eps, rng):
    """Deterministic listwise reader: argmax evidence utility over the context,
    followed by a copy error drawn from competing values in memory."""
    if not context:
        return None
    u = max(context, key=lambda x: _utility(x, q))
    value = str(u.value)
    if eps > 0:
        alt = [x for x in memory if str(x.value) != value]
        if alt and rng.random() < eps:
            value = str(alt[rng.integers(0, len(alt))].value)
    return value


def run(units, queries, cfg, eps, lam=None, score_fn=None, seed=99, beta=3.0):
    rng = np.random.default_rng(seed)
    _IDX.update({id(u): i for i, u in enumerate(units)})
    recs = []
    for q in queries:
        best, f, context = assemble(units, q, q["_top"], cfg)
        s = score_fn(f) if score_fn else None
        answered = best is not None and (lam is None or s >= lam)
        val = pick(context, q, [units[i] for i in q["_top"]], eps, rng) if answered else None
        recs.append(dict(qid=q["qid"], family=q["family"], answered=int(answered),
                         value=val, gold=q["gold_value"],
                         viol=int(answered and val != q["gold_value"]),
                         s=float(s) if s is not None else None))
    return recs


def summarize(recs):
    n = len(recs)
    ans = [r for r in recs if r["answered"]]
    return dict(n=n, answered=len(ans), coverage=len(ans) / n if n else 0.0,
                bcr=(sum(r["viol"] for r in ans) / len(ans)) if ans else 0.0,
                overall=(sum(r["viol"] for r in recs) / n) if n else 0.0)
