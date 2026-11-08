# -*- coding: utf-8 -*-
"""LLM self-assessment baselines on vanilla RAG context (M1), DeepSeek-V3.

Baselines (published decision strategies, reimplemented on the same context):
  REFLECT  Self-RAG-style support gate: answer only if the model itself judges
           the reply fully supported by the context.
  VCONF    verbalized confidence (0-100) with a static threshold calibrated on
           the first half of the stream to reach target risk 0.05 (the static
           split-conformal control of Section 4.6).
  SC5      self-consistency: 5 samples at temperature 0.7, answer by majority
           vote, abstain if fewer than 3/5 agree.

Also replays ACI and the static threshold on top of the *existing* M6 logs
(llm_full_M6.jsonl) offline - no extra API calls.
"""
import json, os, re, sys, time, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_run import (KEY, API, OUTDIR, build_queries_prep, extract_number,
                     is_correct, FAMILIES)

_lock = threading.Lock()


def call(messages, temperature=0.0, max_tokens=400, model="deepseek-chat",
         max_retries=5):
    body = json.dumps({"model": model, "temperature": temperature,
                       "max_tokens": max_tokens,
                       "messages": messages}).encode()
    last = None
    for a in range(max_retries):
        try:
            import urllib.request
            req = urllib.request.Request(API, data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:                              # noqa: BLE001
            last = e
            time.sleep(2 * (a + 1) + np.random.rand())
    raise RuntimeError(f"API failed: {last}")


def parse_json(msg):
    m = re.search(r"\{.*\}", msg, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_one(kind, q, ctx_lines):
    """Returns a record dict. Shared prompt prefix across baselines."""
    user = (f"当前日期：2025年12月1日\n"
            f"客户背景：地区={q['ctx']['region']}；客户类型={q['ctx']['segment']}\n"
            f"客户问题：{q['question']}\n\n【知识资料】\n{ctx_lines}\n")
    gold = q["gold_value"]
    base = dict(qid=q["idx"], family=q["family"], gold=str(gold))

    if kind == "REFLECT":
        user += ("\n请先依据资料回答，再自评：资料是否完全支持你的答案？\n"
                 '只输出 JSON：{"answer": "数值答案（含单位）或 ABSTAIN", '
                 '"support": "full|partial|none"}')
        d = call([{"role": "user", "content": user}])
        j = parse_json(d["choices"][0]["message"]["content"]) or {}
        pred = j.get("answer")
        abstain = (pred is None or str(pred).strip().upper() == "ABSTAIN"
                   or j.get("support") != "full")
        return dict(base, pred=None if abstain else str(pred),
                    usage=d["usage"], kind=kind)

    if kind == "VCONF":
        user += ('\n请回答并在同一 JSON 中给出你对答案正确性的置信度。\n'
                 '只输出 JSON：{"answer": "数值答案（含单位）或 ABSTAIN", '
                 '"confidence": 0到100的整数}')
        d = call([{"role": "user", "content": user}])
        j = parse_json(d["choices"][0]["message"]["content"]) or {}
        pred = j.get("answer")
        conf = j.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        abstain = pred is None or str(pred).strip().upper() == "ABSTAIN"
        return dict(base, pred=None if abstain else str(pred), conf=conf,
                    usage=d["usage"], kind=kind)

    if kind == "SC5":
        user += ('\n只输出 JSON：{"answer": "数值答案（含单位）或 ABSTAIN", '
                 '"reason": "一句话依据"}')
        preds, usage_tot = [], {"prompt_tokens": 0, "completion_tokens": 0}
        for _ in range(5):
            d = call([{"role": "user", "content": user}], temperature=0.7)
            j = parse_json(d["choices"][0]["message"]["content"]) or {}
            p = j.get("answer")
            for k in usage_tot:
                usage_tot[k] += d["usage"][k]
            if p is not None and str(p).strip().upper() != "ABSTAIN":
                preds.append(str(p))
        votes = Counter(extract_number(p) for p in preds)
        votes.pop(None, None)
        if votes and votes.most_common(1)[0][1] >= 3:
            top = votes.most_common(1)[0][0]
            pred = next(p for p in preds if extract_number(p) == top)
            return dict(base, pred=pred, usage=usage_tot, kind=kind)
        return dict(base, pred=None, usage=usage_tot, kind=kind)

    raise ValueError(kind)


def run_baseline(kind, units_prep, tag):
    qs, ctxmap = units_prep
    path = os.path.join(OUTDIR, f"{tag}_{kind}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(l)["qid"] for l in fh if l.strip()}
    todo = [q for q in qs if q["idx"] not in done]
    print(f"[{tag}/{kind}] {len(done)} done, {len(todo)} to go", flush=True)

    def work(q):
        try:
            r = run_one(kind, q, ctxmap[q["idx"]])
        except RuntimeError as e:
            r = dict(qid=q["idx"], family=q["family"], gold=str(q["gold_value"]),
                     pred="ERROR", kind=kind, usage={"prompt_tokens": 0,
                                                     "completion_tokens": 0})
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


def finalize_baseline(kind, tag):
    recs = [json.loads(l) for l in open(
        os.path.join(OUTDIR, f"{tag}_{kind}.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if r.get("pred") != "ERROR"]
    qs = {q["idx"]: q for q in build_queries_prep()[0]}

    if kind == "VCONF":
        order = sorted(recs, key=lambda r: r["qid"])
        half = len(order) // 2
        tr, te = order[:half], order[half:]
        # smallest threshold whose train conditional risk <= 0.05
        best_lam = 1.0
        for lam in np.arange(0, 101, 5):
            a = [r for r in tr if r["conf"] >= lam]
            if a and 1 - np.mean([is_correct(r["pred"], r["gold"])
                                  for r in a]) <= 0.05:
                best_lam = float(lam)
                break
        ev = [r for r in te if r["conf"] >= best_lam]
    else:
        ev = recs
        best_lam = None

    ans = [r for r in ev if r["pred"] is not None]
    s = dict(threshold=best_lam, n=len(recs), answered=len(ans),
             coverage=len(ans) / len(recs),
             bcr=1 - np.mean([is_correct(r["pred"], r["gold"]) for r in ans])
             if ans else 0.0,
             violation=sum(0 if is_correct(r["pred"], r["gold"]) else 1
                           for r in ans) / len(recs))
    by = {}
    for fam in FAMILIES:
        sub = [r for r in ans if r["family"] == fam]
        par = [r for r in recs if r["family"] == fam]
        by[fam] = dict(n=len(par), answered=len(sub),
                       bcr=1 - np.mean([is_correct(r["pred"], r["gold"])
                                        for r in sub]) if sub else 0.0)
    s["by_family"] = by
    print(f"[{kind}] cov={s['coverage']:.3f} bcr={s['bcr']:.3f} "
          f"violation={s['violation']:.3f} thr={best_lam}", flush=True)
    return s


# ---------- offline ACI / static replay on existing M6 LLM logs ----------

def replay_control(tag="llm_full", gamma=0.1, alpha=0.05, lam0=0.5):
    """Replay the answering threshold over the M6 log ordered by reference time.
    The sufficiency scorer is logistic regression over the paper's features,
    fit on the first half of the stream with the LLM's observed outcomes
    (calibration half, reader-independent features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from llm_run import CFG
    from kbc import policies as P
    from kbc.bench import build_kb
    units, T = build_kb()
    qs = build_queries_prep()[0]
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    for q in qs:
        f = P.rrf(bm.scores(q["question"]), dn.scores(q["question"]))
        q["_top"] = list(np.argsort(-f)[:50])
        q["_fs"] = f

    preds = {}
    for l in open(os.path.join(OUTDIR, f"{tag}_M6.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        preds[r["qid"]] = r

    stream = sorted([q for q in qs if q["idx"] in preds],
                    key=lambda q: (q["ctx"]["t_ref"], q["idx"]))
    feats = {}
    for q in stream:
        best, f, cx = P.assemble(units, q, q["_top"], CFG["M6"])
        feats[q["idx"]] = P.vec(f)
    half = len(stream) // 2

    def outcome(q):
        r = preds[q["idx"]]
        return 1 if (r["pred"] is not None and is_correct(r["pred"], r["gold"])) else 0

    X = np.array([feats[q["idx"]] for q in stream[:half]])
    y = np.array([outcome(q) for q in stream[:half]])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    fit = lambda q: float(lr.predict_proba(sc.transform([feats[q["idx"]]]))[0, 1])

    out = {}
    for name in ["static", "aci"]:
        if name == "static":
            lam = 1.0
            for cand in np.arange(0, 1.001, 0.01):
                sel = [outcome(q) for q in stream[:half] if fit(q) >= cand]
                if sel and 1 - np.mean(sel) <= alpha:
                    lam = float(cand)
                    break
        else:
            lam = lam0
        loss_tot, ans_tot, n_tot = 0, 0, 0
        for q in stream[half:]:
            n_tot += 1
            if fit(q) >= lam:
                r = preds[q["idx"]]
                if r["pred"] is None:
                    continue          # released to pipeline, but LLM abstained
                ans_tot += 1
                loss = 1 - outcome(q)
                loss_tot += loss
                if name == "aci":
                    lam = min(1.0, max(0.0, lam + gamma * (loss - alpha)))
        out[name] = dict(gamma=gamma, alpha=alpha, lam_final=round(float(lam), 3),
                         n=n_tot, coverage=ans_tot / n_tot,
                         bcr=loss_tot / ans_tot if ans_tot else 0.0,
                         harm=loss_tot / n_tot)
        print(f"[replay/{name}] {out[name]}", flush=True)
    return out


def main():
    kinds = sys.argv[1:] or ["REFLECT", "VCONF", "SC5"]
    prep = build_queries_prep()
    out = {}
    for k in kinds:
        if k == "REPLAY":
            out["replay"] = replay_control()
            continue
        run_baseline(k, prep, "base")
        out[k] = finalize_baseline(k, "base")
    path = os.path.join(OUTDIR, "base_summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    json.dump(old, open(path, "w"), ensure_ascii=False, indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
