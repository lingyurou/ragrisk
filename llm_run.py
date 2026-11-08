# -*- coding: utf-8 -*-
"""LLM replication harness: replace the deterministic reader with a real LLM.

Configs (context assembled by the same decision layer as the paper):
  M1  vanilla RAG   : top-50 retrieved units, no metadata handling
  M5  filtered      : slot + temporal(as-of) + scope filtered candidates
  M6  BIND-RAG      : single normatively resolved unit

Usage: python llm_run.py --pilot          # 30 queries x 3 configs
       python llm_run.py --full M1 M5 M6  # full run, resumable
"""
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbc.bench import build_kb, build_queries, FAMILIES
from kbc import policies as P

API = os.environ.get("LLM_API_BASE", "https://api.deepseek.com/chat/completions")
KEY = os.environ.get("LLM_API_KEY")
if not KEY:                                    # optional local fallback, not shipped
    _f = os.path.join(os.path.dirname(__file__), ".dskey")
    if os.path.exists(_f):
        KEY = open(_f).read().strip()
    else:
        raise SystemExit("LLM_API_KEY is not set; export your API key before running.")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
NOW_LABEL = "2025年12月1日"          # benchmark month 35 -> calendar date
OUTDIR = os.path.join(os.path.dirname(__file__), "llm_results")
os.makedirs(OUTDIR, exist_ok=True)

CFG = {
    "M1": {},
    "M5": dict(slot=True, temporal="ref", scope=True),
    "M6": dict(slot=True, temporal="ref", scope=True, resolve=True),
}

SYSTEM = (
    "你是银行的合规知识问答引擎。你只能依据提供的【知识资料】回答客户问题，"
    "禁止使用你自带的任何知识；资料之间若存在版本、层级或适用范围差异，"
    "请依据资料自身的生效日期、适用范围和效力措辞判断应执行哪一条。"
    "若资料不足以唯一确定答案，必须回答 ABSTAIN。只输出一个 JSON 对象。"
)

SYS_PROMPT = (
    "你是银行的合规知识问答引擎。只能依据【知识资料】回答，禁止使用自带知识。"
    "资料间存在版本/层级/适用范围差异时，依据资料自身的生效日期、适用范围与效力措辞"
    "判断应执行哪一条。不足以唯一确定答案时必须输出 ABSTAIN。只输出一个 JSON 对象。"
)


def render(u):
    return u.text


def make_prompt(q, context):
    lines = [f"当前日期：{NOW_LABEL}",
             f"客户背景：地区={q['ctx']['region']}；客户类型={q['ctx']['segment']}",
             f"客户问题：{q['question']}", "", "【知识资料】"]
    if not context:
        lines.append("（无）")
    else:
        for i, u in enumerate(context, 1):
            lines.append(f"[{i}] {render(u)}")
    lines += ["", '请只输出 JSON：{"answer": "数值答案（含单位）或 ABSTAIN", '
                  '"unit_id": "所依据资料的编号（如 [3]），弃答填 null", '
                  '"reason": "一句话依据"}']
    return "\n".join(lines)


def call_llm(prompt, temperature=0.0, max_retries=5):
    if MODEL == "deepseek-reasoner":   # reasoner: no system role, no temp
        msgs = [{"role": "user", "content": SYS_PROMPT + "\n\n" + prompt}]
        payload = {"model": MODEL, "max_tokens": 2000, "messages": msgs}
    else:
        payload = {"model": MODEL, "temperature": temperature, "max_tokens": 300,
                   "messages": [{"role": "system", "content": SYS_PROMPT},
                                {"role": "user", "content": prompt}]}
    body = json.dumps(payload).encode()
    last = None
    for a in range(max_retries):
        try:
            req = urllib.request.Request(API, data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]["content"]
            return msg, d["usage"]
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(2 * (a + 1) + np.random.rand())
    raise RuntimeError(f"API failed after retries: {last}")


def parse_reply(msg):
    m = re.search(r"\{.*\}", msg, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d
    except json.JSONDecodeError:
        return None


def extract_number(s):
    if not s:
        return None
    s = str(s).replace(",", "").replace("，", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    a, b = extract_number(pred), extract_number(str(gold))
    if a is None or b is None:
        return str(pred).strip() == str(gold).strip()
    return abs(a - b) <= 0.005 * max(1.0, abs(b))


def evaluate_query(units, q, cfg_name, fit=None):
    cfg = CFG[cfg_name]
    best, f, context = P.assemble(units, q, q["_top"], cfg)
    prompt = make_prompt(q, context)
    msg, usage = call_llm(prompt)
    d = parse_reply(msg)
    pred = None if d is None else d.get("answer")
    abstain = pred is None or str(pred).strip().upper() == "ABSTAIN" or d is None
    gold = q["gold_value"]
    rec = dict(cfg=cfg_name, qid=q["idx"], family=q["family"],
               question=q["question"], gold=str(gold),
               pred=None if abstain else str(pred),
               unit_id=None if d is None else d.get("unit_id"),
               reason=None if d is None else d.get("reason"),
               n_ctx=len(context),
               correct=(not abstain) and is_correct(pred, gold),
               usage=usage, raw=msg[:400])
    return rec


def summarize(recs):
    ans = [r for r in recs if r["pred"] is not None]
    cov = len(ans) / len(recs)
    bcr = 1 - np.mean([r["correct"] for r in ans]) if ans else 0.0
    harm = 1 - np.mean([r["correct"] for r in recs])
    by = {}
    for fam in FAMILIES:
        sub = [r for r in recs if r["family"] == fam]
        a = [r for r in sub if r["pred"] is not None]
        by[fam] = dict(n=len(sub), coverage=len(a) / len(sub) if sub else 0,
                       bcr=1 - np.mean([r["correct"] for r in a]) if a else 0.0)
    return dict(n=len(recs), coverage=cov, bcr=bcr, harm=harm, by_family=by)


_lock = threading.Lock()


def run_config(units, queries, cfg_name, tag):
    path = os.path.join(OUTDIR, f"{tag}_{cfg_name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(l)["qid"] for l in fh if l.strip()}
    todo = [q for q in queries if q["idx"] not in done]
    print(f"[{tag}/{cfg_name}] {len(done)} done, {len(todo)} to go", flush=True)

    def work(q):
        try:
            r = evaluate_query(units, q, cfg_name)
        except RuntimeError as e:
            r = dict(cfg=cfg_name, qid=q["idx"], family=q["family"],
                     question=q["question"], gold=str(q["gold_value"]),
                     pred="ERROR", error=str(e)[:200], correct=False,
                     usage={"prompt_tokens": 0, "completion_tokens": 0})
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    s = summarize(recs)
    pt = sum(r["usage"]["prompt_tokens"] for r in recs)
    ct = sum(r["usage"]["completion_tokens"] for r in recs)
    print(f"[{tag}/{cfg_name}] cov={s['coverage']:.3f} bcr={s['bcr']:.3f} "
          f"harm={s['harm']:.3f}  tokens in={pt} out={ct}", flush=True)
    return s


NOW = 35   # benchmark "current" month -> 2025-12-01


def build_queries_prep():
    """Shared prep: corpus, temporal coercion, retrieval top-50, and the
    rendered M1 (vanilla top-50) context lines per query."""
    from kbc.bench import build_kb as _bk
    units, T = _bk()
    coerce_now(units, qs := build_queries(units, T))
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    ctxmap = {}
    for q in qs:
        f = P.rrf(bm.scores(q["question"]), dn.scores(q["question"]))
        q["_top"] = list(np.argsort(-f)[:50])
        q["_fs"] = f
        _, _, context = P.assemble(units, q, q["_top"], {})   # M1: no filters
        ctxmap[q["idx"]] = "\n".join(f"[{i}] {u.text}"
                                     for i, u in enumerate(context, 1)) \
            if context else "（无）"
    return qs, ctxmap


def coerce_now(units, qs):
    """Temporal coherence for the LLM protocol: availability-type questions say
    "目前/现在", so bind t_ref to NOW and recompute gold. VINTAGE keeps its
    explicit contract date; GAP stays uncovered."""
    import kbc.bench as B
    n = 0
    for q in qs:
        if q["family"] in ("VINTAGE", "GAP"):
            continue
        ctx = dict(q["ctx"], t_ref=NOW)
        gold = B.resolve(units, ctx)[0]
        if not gold:
            continue
        q["ctx"] = ctx
        q["gold_value"] = str(gold[0].value)
        q["gold_units"] = [u.uid for u in gold]
        q["gold_tier"] = gold[0].tier
        n += 1
    print(f"coerced t_ref=NOW for {n}/{len(qs)} queries", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", nargs="*", default=None)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    units, T = build_kb()
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    qs = build_queries(units, T)
    coerce_now(units, qs)
    for q in qs:
        f = P.rrf(bm.scores(q["question"]), dn.scores(q["question"]))
        q["_top"] = list(np.argsort(-f)[:50])
        q["_fs"] = f

    if a.pilot:
        rng = np.random.default_rng(7)
        pilot = []
        for fam in FAMILIES:
            sub = [q for q in qs if q["family"] == fam]
            k = max(1, round(30 * len(sub) / len(qs)))
            pilot += [sub[i] for i in rng.permutation(len(sub))[:k]]
        pilot = pilot[:30]
        print("pilot families:", {f: sum(1 for q in pilot if q['family'] == f)
                                  for f in FAMILIES}, flush=True)
        out = {}
        for c in ["M1", "M5", "M6"]:
            out[c] = run_config(units, pilot, c, a.tag + "_pilot")
        json.dump(out, open(os.path.join(OUTDIR, f"{a.tag}_pilot_summary.json"), "w"),
                  ensure_ascii=False, indent=1)
    else:
        out = {}
        for c in (a.full or ["M1", "M5", "M6"]):
            out[c] = run_config(units, qs, c, a.tag)
        json.dump(out, open(os.path.join(OUTDIR, f"{a.tag}_summary.json"), "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
