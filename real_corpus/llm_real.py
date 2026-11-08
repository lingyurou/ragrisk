# -*- coding: utf-8 -*-
"""LLM validation on the real-instrument corpus (BankKB-R).
Configs: M1 vanilla top-50 context vs M6 BIND-RAG resolved single-unit context.
Usage: python llm_real.py --pilot | --full M1 M6
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from kbc import policies as P
from real_kb import build_real_kb, NOW_R
from run_real import build_queries, assemble_real, cn_date
from llm_run import call_llm, parse_reply, extract_number, OUTDIR, KEY

NOW_LABEL = cn_date(NOW_R - 3) + "1日"       # FACT questions anchor 2026-07-01


def norm(s):
    s = re.sub(r"[，。；：、（）()\[\]【】《》“”\"'：,.;:%％\u3000\s]", "", str(s))
    return s


def bigrams(s):
    return set(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else {s}


JUDGE_SYS = ("你是合规答案评分器。给定【客户问题】【金标准】（由规章条款提炼的标准答案要点）与【模型答案】，"
             "判断模型答案在规范内容上是否与金标准一致："
             "1. 模型答案覆盖金标准要点即算一致，允许措辞不同、更详细或更简洁；"
             "模型答案含金标准之外的额外内容时，只要额外内容与金标准不矛盾，仍算一致。"
             "2. 是非问上的简短结论（如是/否/能/不能/不合规），方向与金标准一致即算一致。"
             "3. 金标准含数值时数值必须一致，单位换算后相等算一致"
             "（如5年=60期按年折算、1.2万元=12000元、0.5小时=30分钟）。"
             "4. 模型答案的规范方向与金标准相反，或遗漏导致客户会做出与金标准不同的行动"
             "（缴不同的费、拿不同的额度、误以为被允许或被禁止），判不一致。"
             "5. 若金标准为“无法确定/ABSTAIN”类而模型给出了明确答案，判不一致。"
             "工作方法：第一步，从金标准推导出对【客户问题】的直接正确回答（一句话）；"
             "第二步，只将模型答案与推导出的回答比较，不与金标准原文逐字比对；"
             "模型答案未复述背景细节（解释、渠道清单、日期出处）或措辞比金标准简短，不算不一致。"
             "只输出 JSON：{\"equivalent\": true/false, \"why\": \"一句话\"}")


def judge_call(pred, gold, question=""):
    import urllib.request
    payload = {"model": "deepseek-chat", "temperature": 0.0, "max_tokens": 120,
               "messages": [{"role": "system", "content": JUDGE_SYS},
                            {"role": "user", "content":
                             f"【客户问题】{question}\n【金标准】{gold}\n【模型答案】{pred}"}]}
    body = json.dumps(payload).encode()
    for a in range(4):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions", data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            m = parse_reply(d["choices"][0]["message"]["content"])
            return bool(m and m.get("equivalent"))
        except Exception:                                    # noqa: BLE001
            time.sleep(2 * (a + 1))
    return None                                             # judge unavailable


def is_correct_real(pred, gold, question=""):
    """Numeric golds -> tolerance; text golds -> containment or bigram F1 >= 0.5."""
    if pred is None or gold is None:
        return False
    a, b = extract_number(pred), extract_number(str(gold))
    g = str(gold)
    if extract_number(g) is not None and len(norm(g)) <= 12:
        if a is not None and abs(a - b) <= 0.005 * max(1.0, abs(b)):
            return True
        return bool(judge_call(pred, g, question))   # unit conversion / phrasing -> judge
    pa, pg = norm(pred), norm(g)
    if not pg:
        return False
    if pg in pa or pa in pg:
        return True
    A, G = bigrams(pg), bigrams(pa)
    if A and 2 * len(A & G) / (len(A) + len(G)) >= 0.5:
        return True
    return bool(judge_call(pred, g, question))


def make_prompt_real(q, context, cfg_name="M1"):
    lines = [f"当前日期：{NOW_LABEL}",
             f"客户背景：地区={q['ctx']['region']}；客户类型={q['ctx']['segment']}",
             f"客户问题：{q['question']}", "", "【知识资料】"]
    if not context:
        lines.append("（无）")
    else:
        for i, u in enumerate(context, 1):
            lines.append(f"[{i}] {u.text}")
    lines += ["", '请只输出 JSON：{"answer": "按现行有效规定应执行的标准或结论（含单位；'
                  '资料不足以唯一确定时输出 ABSTAIN）", '
                  '"unit_id": "所依据资料编号（如 [3]），弃答填 null", "reason": "一句话依据"}']
    if cfg_name == "M6":
        lines += ["说明：以上资料是决策层按效力层级、生效日期与适用范围解析出的应执行条款，"
                  "请直接依据该条款作答；若该条款仍不足以回答客户问题，输出 ABSTAIN。"]
    return "\n".join(lines)


def evaluate_query(units, q, cfg_name):
    cfg = {"M1": {}, "M6": dict(slot=True, temporal="ref", scope=True, resolve=True)}[cfg_name]
    best, f, context = assemble_real(units, q, q["_top"], cfg)
    msg, usage = call_llm(make_prompt_real(q, context, cfg_name))
    d = parse_reply(msg)
    pred = None if d is None else d.get("answer")
    abstain = pred is None or str(pred).strip().upper() == "ABSTAIN"
    gold = q.get("gold_disp", q["gold_value"])
    rec = dict(cfg=cfg_name, qid=q["qid"], family=q["family"], question=q["question"],
               gold=str(gold), pred=None if abstain else str(pred),
               n_ctx=len(context), unit_id=None if d is None else d.get("unit_id"),
               reason=None if d is None else d.get("reason"),
               answered=int(not abstain),
               viol=int((not abstain) and not is_correct_real(pred, gold, q["question"])),
               usage=usage, raw=msg[:300])
    return rec


_lock = threading.Lock()


def run_config(units, queries, cfg_name, tag):
    path = os.path.join(OUTDIR, f"{tag}_{cfg_name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(l)["qid"] for l in fh if l.strip()}
    todo = [q for q in queries if q["qid"] not in done]
    print(f"[{tag}/{cfg_name}] {len(done)} done, {len(todo)} to go", flush=True)

    def work(q):
        try:
            r = evaluate_query(units, q, cfg_name)
        except RuntimeError as e:
            r = dict(cfg=cfg_name, qid=q["qid"], family=q["family"], question=q["question"],
                     gold=str(q["gold_value"]), pred="ERROR", answered=0, viol=0, error=str(e)[:200],
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
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    recs = [r for r in recs if r.get("pred") != "ERROR"]
    return summarize(recs)


def summarize(recs):
    n = len(recs)
    ans = [r for r in recs if r["answered"]]
    fams = ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]
    by = {}
    for fam in fams:
        sub = [r for r in recs if r["family"] == fam]
        a = [r for r in sub if r["answered"]]
        by[fam] = dict(n=len(sub), answered=len(a),
                       coverage=len(a) / len(sub) if sub else 0.0,
                       bcr=(sum(r["viol"] for r in a) / len(a)) if a else 0.0)
    return dict(n=n, answered=len(ans), coverage=len(ans) / n if n else 0.0,
                bcr=(sum(r["viol"] for r in ans) / len(ans)) if ans else 0.0,
                overall=sum(r["viol"] for r in recs) / n if n else 0.0,
                by_family=by)


def prep():
    units = build_real_kb()
    qs = build_queries(units)
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    for q in qs:
        f = P.rrf(bm.scores(q["question"]), dn.scores(q["question"]))
        q["_top"], q["_fs"] = list(np.argsort(-f)[:50]), f
    return units, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", nargs="*", default=["M1", "M6"])
    ap.add_argument("--tag", default="real")
    a = ap.parse_args()
    units, qs = prep()
    if a.pilot:
        rng = np.random.default_rng(7)
        fams = ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]
        pilot = []
        for fam in fams:
            sub = [q for q in qs if q["family"] == fam]
            k = max(1, round(28 * len(sub) / len(qs)))
            pilot += [sub[i] for i in rng.permutation(len(sub))[:k]]
        pilot = pilot[:28]
        out = {}
        for c in ["M1", "M6"]:
            out[c] = run_config(units, pilot, c, a.tag + "_pilot")
        json.dump(out, open(os.path.join(OUTDIR, f"{a.tag}_pilot_summary.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1)[:1500])
    else:
        out = {}
        for c in a.full:
            out[c] = run_config(units, qs, c, a.tag)
        json.dump(out, open(os.path.join(OUTDIR, f"{a.tag}_summary.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
