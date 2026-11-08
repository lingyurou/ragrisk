# BIND-RAG — reference code and data

Code and data accompanying the manuscript *BIND-RAG: binding-grounded answering over
regulated knowledge bases* (submitted to **Knowledge-Based Systems**).

The paper's claim is that in a regulated corpus most answer errors are not retrieval
errors but **normative selection** errors: the passage retrieved is relevant, but it is
the wrong *binding* provision (superseded, out of scope, or overridden by a
higher-authority rule). This repository contains the benchmark generator, the ladder of
retrieval/decision baselines, the risk controller, the real-instrument case corpus, and
every script that produces a number in the paper.

## Layout

| Path | What it is | Paper |
|---|---|---|
| `kbc/bench.py` | **BankKB-C** generator: 756 queries, bitemporal versions, scope/authority strata, corruption model. Deterministic given `SEED`; gold binding sets are exact by construction, not annotated. | §6, §7.1 |
| `kbc/policies.py` | Five-rule normative resolver (admissibility, *lex specialis*, *lex stringentior*, authority, recency) and the answer/abstain policies. | Def. 1–4, Alg. 1 |
| `run_experiments.py` | Main driver: ladder M1–M6 + ACI controller, ablations, hyper-parameters, corruption sweeps. Writes `results.json`. | §7.2–7.8, Tables 3–7 |
| `seed_variance.py` | Same head-line numbers over independent 50/50 splits of the 756-query set → mean ± sd (defence against "the gain is inside the noise"). | §7.2 |
| `llm_baselines.py` | Published self-assessment gates reimplemented on one shared context (M1): Self-RAG-style support check, verbalized confidence, self-consistency; plus offline ACI/threshold replay on stored M6 logs. | §7.9 |
| `figures.py` | Hand-rolled SVG figures from `results.json` (no plotting dependency). | Figs. 1–5 |
| `llm_run.py` | DeepSeek reader over the same ladder on BankKB-C. | §7.10 |
| `llm_results/` | Stored per-question records (`.jsonl`) and summaries (`.json`) for every LLM run — pilot, full, re-scored. | §7.10, Table 8 |
| `real_corpus/` | **BankKB-R**: the real-instrument case study (15 public documents, 105 queries). | §7.11 |
| `real_corpus/fetch_docs.py` | Re-fetches the 15 documents into `real_corpus/raw/` and verifies their `sha256` against `manifest.json`. | §7.11 |
| `real_corpus/real_kb.py` | Unit base for the real corpus. Every unit carries a verbatim `quote` that must be a substring of the fetched document text (`verify()` enforces it), so no provision text is invented. | §7.11 |
| `real_corpus/run_real.py` | BankKB-R ladder with the real-corpus resolver. Writes `results_real.json`. | §7.11 |
| `real_corpus/llm_real.py` | DeepSeek reader + judge on BankKB-R. | §7.11 |
| `real_corpus/llm_real_oai.py` | GPT-4o-mini reader on BankKB-R with a **fixed independent GPT-4o judge** (cross-vendor robustness). | §7.11, Table 8 |
| `real_corpus/rescore.py` | Re-judges stored `.jsonl` records offline with the current judge — no reader calls, so re-scoring cannot move the reader's answers. | §7.11 |
| `real_corpus/judge_audit.log` | Manual audit of 20 judge decisions (14 overturned, 6 confirmed); the basis for the judge protocol described in the paper. | §7.11 |

## Quickstart — everything that needs no API key

```bash
python3 -m pip install -r requirements.txt

python3 run_experiments.py      # -> results.json   (main tables, one CPU core, a few minutes)
python3 seed_variance.py 5      # -> seed_variance.json  (~30 s per split)
python3 figures.py              # -> figures/*.svg
```

Expected head-line values (`results.json`, corruption ρ = 0.15, α = 0.05, seed 2026):

| | binding-consistent rate (BCR) |
|---|---|
| M1 vanilla RAG | 0.783 |
| M4 reference-time retrieval | 0.718 |
| M6 BIND-RAG (full resolver) | 0.199 |
| M7 BIND-RAG + ACI controller | harm 0.058 @ coverage 0.550 |

`seed_variance.py 5` gives M1 = 0.792 ± 0.013 and M6 = 0.216 ± 0.013 over five
independent splits: the 0.58 gap is about forty times its split-to-split standard
deviation.

## LLM experiments (needs credentials)

The LLM scripts take the endpoint from the environment; **no key is stored in this
repository**.

```bash
export LLM_API_KEY=...                       # required
export LLM_API_BASE=https://api.deepseek.com/chat/completions   # default; any OpenAI-compatible endpoint
export LLM_MODEL=deepseek-chat                                   # default

python3 llm_run.py --pilot          # BankKB-C, small pilot first
python3 llm_run.py --full M1 M5 M6  # BankKB-C, full run
python3 llm_baselines.py            # self-assessment baselines on M1
```

BankKB-R case study (vendor A = DeepSeek, vendor B = OpenAI):

```bash
cd real_corpus
python3 run_real.py                       # resolver-only ladder, no API calls
python3 llm_real.py --pilot               # DeepSeek reader + judge
python3 llm_real_oai.py --pilot           # GPT-4o-mini reader + GPT-4o judge
python3 llm_real_oai.py --full M1 M6
python3 rescore.py --provider oai --files ../llm_results/oai_M1.jsonl
```

Total cost of the LLM experiments reported in the paper was about CNY 23.

## Data and integrity notes

* **BankKB-C is semi-synthetic and deterministic.** `build_kb` / `build_queries` are
  seeded; gold labels come from the generator, so the binding set is exact rather than
  human-annotated. Every table, ablation and sweep in the paper is reproduced
  bit-for-bit on the same `numpy` / `scikit-learn` versions. The single exception is
  the qualitative Fig. 2 risk–coverage curve, which depends on the per-query draw of
  the ε-greedy answer sampler (ε = 0.05): reseeding that draw moves the curve by
  ≤ 2 points once there are more than ~40 queries on the curve (≤ 9 points among the
  first few dozen, where one query moves the rate by 1/n), with the ordering of the
  four policies unchanged.
* **BankKB-R is real and checkable.** The 15 documents are public (State Council
  regulations, PBoC and regulator notices, bank tariff pages). `manifest.json` records
  `url`, `bytes`, `text_chars`, `fetched_at` and `sha256` for each; `fetch_docs.py`
  re-downloads and re-verifies them, and `real_kb.verify()` refuses any unit whose
  `quote` is not a verbatim substring of the fetched text.
* **Reader ≠ judge for the cross-vendor run.** DeepSeek runs use the reader as judge;
  the GPT-4o-mini run uses a fixed GPT-4o judge, because a reader-model judge showed a
  systematic completeness bias (it marks short, directionally correct answers as
  inconsistent). `judge_audit.log` lists the 20 audited decisions.
* Small-sample honesty: BankKB-R has 105 queries, so all rates are reported with 95%
  Wilson intervals, and the per-question comparison uses an exact McNemar test
  (p = 0.22 — directionally consistent, not individually significant at this size).

## Changes made for this release

* Absolute paths of the authoring machine were replaced by script-relative paths;
  numerics are unchanged (verified: the relocated copy reproduces the published
  `results.json`).
* `run_experiments.py` used `hash(qid)` to seed the ε-greedy sampler, which Python
  randomises per process (string hash salt), so Fig. 2's curve was not reproducible
  across runs. It now uses `zlib.crc32(qid)`, and repeated runs produce identical
  `results.json`. The curve in the manuscript was drawn from one draw of the old
  seeding and differs from the shipped file within the tolerance stated above.
* Local credential files (`.dskey`, `real_corpus/.openai_key`) are not included;
  `LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL` are read from the environment.
* The `llm_real_oai.py` header comment was corrected to describe the fixed independent
  judge actually used in the paper.

## Contact

Corresponding author — see the manuscript. Issues and questions about reproducing the
results are welcome via the issue tracker.
