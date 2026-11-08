# -*- coding: utf-8 -*-
"""OpenAI (gpt-4o-mini) validation on BankKB-R, same protocol as llm_real.py,
except that the judge is a fixed independent model (gpt-4o) rather than the
reader model, so that a reader-side verbosity or completeness bias cannot
inflate its own score.
Usage: python llm_real_oai.py --pilot | --full M1 M6
Env:   LLM_API_BASE / LLM_API_KEY / LLM_MODEL   (set by caller)
"""
import json, time

import llm_real
from llm_run import parse_reply


def judge_call_oai(pred, gold, question=""):
    import urllib.request, os
    jmodel = os.environ.get("JUDGE_MODEL", "gpt-4o")   # fixed independent judge
    payload = {"model": jmodel,
               "temperature": 0.0, "max_tokens": 150,
               "messages": [{"role": "system", "content": llm_real.JUDGE_SYS},
                            {"role": "user", "content":
                             f"【客户问题】{question}\n【金标准】{gold}\n【模型答案】{pred}"}]}
    body = json.dumps(payload).encode()
    last = None
    for a in range(6):                                   # TLS to OpenAI is flaky here
        try:
            req = urllib.request.Request(
                os.environ["LLM_API_BASE"], data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            m = parse_reply(d["choices"][0]["message"]["content"])
            if m is None or "equivalent" not in m:
                last = RuntimeError(f"unparseable judge reply: {str(m)[:80]}")
                time.sleep(2 * (a + 1))
                continue
            return bool(m.get("equivalent"))
        except Exception as e:                           # noqa: BLE001
            last = e
            time.sleep(min(60, 3 * (2 ** a)))
    raise RuntimeError(f"judge unavailable after retries: {last}")


llm_real.judge_call = judge_call_oai
llm_real.is_correct_real.__globals__["judge_call"] = judge_call_oai

if __name__ == "__main__":
    import sys
    sys.argv[0] = "llm_real_oai.py"
    llm_real.main()
