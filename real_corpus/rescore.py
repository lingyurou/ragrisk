# -*- coding: utf-8 -*-
"""Re-score stored jsonl records with the current judge, without re-calling
the reader. Usage:
  python rescore.py --provider ds  --files ../llm_results/real2_M1.jsonl ...
  python rescore.py --provider oai --files ../llm_results/oai_M1.jsonl ...
Writes <file> in place and prints the recomputed summary.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llm_real

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["ds", "oai"], required=True)
ap.add_argument("--files", nargs="+", required=True)
a = ap.parse_args()

if a.provider == "oai":
    import llm_real_oai
    llm_real.judge_call = llm_real_oai.judge_call_oai
else:  # deepseek: llm_real.judge_call is already deepseek-based (KEY from .dskey)
    pass

for path in a.files:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n_re = 0
    for p in range(5):                                   # retry passes for judge flakiness
        pending = []
        for r in recs:
            if not r.get("answered") or not r.get("pred"):
                r["viol"] = 0
                continue
            try:
                ok = llm_real.is_correct_real(r["pred"], r["gold"], r.get("question", ""))
            except RuntimeError:
                pending.append(r)
                continue
            v = int(not ok)
            if v != r.get("viol"):
                n_re += 1
            r["viol"] = v
        if not pending:
            break
        print(f"  {os.path.basename(path)} pass{p+1}: {len(pending)} judge retries",
              flush=True)
    else:
        raise SystemExit(f"judge still failing for {len(pending)} records in {path}")
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = llm_real.summarize(recs)
    print(f"{os.path.basename(path)}: flipped={n_re} cov={s['coverage']:.3f} "
          f"bcr={s['bcr']:.3f} overall={s['overall']:.3f}", flush=True)
    print("  by_family:", {f: (v["coverage"], round(v["bcr"], 3))
                           for f, v in s["by_family"].items() if v["n"]})
