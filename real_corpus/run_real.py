# -*- coding: utf-8 -*-
"""Case study driver: BIND-RAG ladder on the real-instrument corpus (BankKB-R).

Same ladder semantics as run_experiments.py but with the real-corpus resolver
(resolve_real) and a reader whose "current" pointer is the build month (NOW_R).
Outputs results_real.json.
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kbc import policies as P                      # BM25 / Dense / rrf reuse
from kbc.bench import Unit
from real_kb import (build_real_kb, resolve_real, laxer_real, m, NOW_R,
                     MODE, TIER_RANK, OPERATIVE_REAL)

OUT = os.path.join(HERE, "results_real.json")
K = 50
TIERS_DESC = {"L1": "法律/行政法规", "L2": "部门规章/规范性文件", "L5": "营销宣传/FAQ"}


def cn_date(tm):
    return f"{2014 + tm // 12}年{tm % 12 + 1}月"


# ------------------------------------------------------------------ queries ----
Q_OVERRIDES = {
    ("大额存单", "起存金额"): "大额存单个人起存金额是多少？",
    ("公募理财", "销售起点"): "在银行买公募理财，销售起点是多少？",
    ("公募理财", "保本承诺"): "银行理财可以承诺保本保收益吗？",
    ("公募理财", "预期收益率宣传"): "理财产品宣传时可以写预期收益率吗？",
    ("公募理财", "代销渠道"): "公募理财可以通过哪些渠道销售？",
    ("公募理财", "登记编码"): "怎么辨别一款理财产品是正规登记过的？",
    ("公募理财", "风险匹配"): "我的风险测评是二级，能买四级理财吗？",
    ("公募理财", "专区销售"): "买理财为什么要在销售专区录音录像？",
    ("公募理财", "开放式理财杠杆"): "开放式公募理财的杠杆水平上限是多少？",
    ("公募理财", "封闭式理财杠杆"): "封闭式公募理财的杠杆水平上限是多少？",
    ("公募理财", "私募固收起点"): "私募固收类理财的起投金额是多少？",
    ("公募理财", "私募混合起点"): "私募混合类理财的起投金额是多少？",
    ("公募理财", "私募权益起点"): "私募权益类理财的起投金额是多少？",
    ("公募理财", "资金池运作"): "银行可以拿理财资金做资金池运作吗？",
    ("大额存单", "期限品种"): "大额存单有哪些期限？",
    ("大额存单", "可质押性"): "大额存单可以质押贷款吗？",
    ("大额存单", "付息方式"): "大额存单的付息方式有哪几种？",
    ("大额存单", "转让与提前支取"): "大额存单能不能转让或提前支取？",
    ("大额存单", "发行条款披露"): "银行发行大额存单前要披露发行条款吗？",
    ("结构性存款", "销售起点"): "结构性存款的销售起点是多少？",
    ("结构性存款", "投资冷静期"): "买结构性存款有投资冷静期吗？",
    ("结构性存款", "风险揭示语"): "结构性存款销售时必须做哪些风险揭示？",
    ("结构性存款", "误导销售"): "结构性存款可以当普通存款卖吗？",
    ("结构性存款", "存款保险保费缴纳"): "结构性存款要交存款保险保费吗？",
    ("结构性存款", "市场风险资本上限"): "结构性存款的市场风险资本上限是多少？",
    ("结构性存款", "发行报告披露"): "结构性存款成立后多久披露发行报告？",
    ("结构性存款", "产品账单"): "结构性存款存续期间多久能拿到一次产品账单？",
    ("个人住房贷款", "年化利率展示"): "贷款营销时必须展示年化利率吗？",
    ("个人住房贷款", "成本口径"): "贷款年化利率要把哪些成本算进去？",
    ("个人住房贷款", "日月利率展示"): "贷款宣传只写日利率不写年化利率合规吗？",
    ("个人住房贷款", "计算方法"): "贷款年化利率用单利计算有什么要求？",
    ("个人住房贷款", "适用机构"): "个人住房贷款营销展示年化利率的要求，适用于哪些机构？",
    ("信用卡分期", "分期期限上限"): "信用卡分期最长可以分多少年？",
    ("信用卡分期", "预借现金分期额度"): "信用卡预借现金分期最多能办多少额度？",
    ("信用卡分期", "预借现金分期期限"): "信用卡预借现金分期最长几年？",
    ("信用卡分期", "成本展示形式"): "分期业务成本要用利息还是手续费展示？",
    ("信用卡分期", "年化展示"): "分期合同里必须展示年化利率吗？",
    ("信用卡分期", "提前结清计息"): "信用卡分期提前结清，利息怎么算？",
    ("信用卡分期", "二次分期"): "已经分期的部分还能再办分期吗？",
    ("信用卡分期", "最低还款额"): "分期业务可以提供最低还款额服务吗？",
    ("信用卡分期", "睡眠卡比例"): "长期睡眠信用卡占比不能超过多少？",
    ("互联网存款", "服务区域"): "地方银行的存款产品能卖给全国客户吗？",
    ("互联网存款", "非自营平台"): "银行能在第三方平台上卖定期存款吗？",
    ("互联网存款", "存款保险营销"): "银行可以用存款保险做营销宣传吗？",
    ("存款保险", "偿付限额"): "存款保险最多赔多少？",
    ("存款保险", "偿付时限"): "存款保险赔付需要多久？",
    ("存款保险", "限额内偿付"): "50万以内的存款怎么赔付？",
    ("个人养老金", "年度缴费上限"): "个人养老金每年最多能缴多少？",
    ("个人养老金", "产品范围"): "个人养老金账户里的钱可以买哪些产品？",
    ("个人养老金", "领取条件"): "什么情况下可以领个人养老金？",
    ("个人养老金", "实施范围"): "个人养老金现在全国都能参加了吗？",
    ("个人养老金", "账户变更"): "个人养老金资金账户可以换银行吗？",
    ("个人养老金", "税收优惠范围"): "个人养老金税收优惠全国都适用吗？",
    ("个人养老金", "超额缴费处理"): "个人养老金缴费超过上限怎么办？",
    ("个人养老金", "账户继承"): "人不在了，个人养老金账户里的钱怎么办？",
    ("个人养老金", "未投资资金计息"): "个人养老金账户里没买产品的钱有利息吗？",
    ("个人养老金", "销售适当性"): "银行可以随便给我推荐个人养老金产品吗？",
    ("个人养老金", "默认投资"): "个人养老金有默认投资服务吗？",
    ("个人养老金", "资金账户唯一性"): "个人养老金资金账户可以开几个？",
    ("个人养老金", "账户封闭运行"): "个人养老金账户平时能取钱吗？",
}

Q_EXTRA = []   # hand-authored non-FACT queries, filled below
def q(family, product, attr, t, question, region="全部", segment="全部客户"):
    Q_EXTRA.append(dict(family=family, product=product, attr=attr, region=region,
                        segment=segment, t_ref=t, question=question))

# ---- VINTAGE: explicit contract/purchase dates before a revision -------------
q("VINTAGE", "大额存单", "起存金额", m(2015, 9),
  f"我是{cn_date(m(2015,9))}买的大额存单，当时个人起存金额是多少？请按购买当时有效的规定回答。")
q("VINTAGE", "大额存单", "起存金额", m(2016, 3),
  f"我是{cn_date(m(2016,3))}签约的大额存单客户，当时的起存门槛是多少？")
q("VINTAGE", "公募理财", "销售起点", m(2016, 7),
  f"我是{cn_date(m(2016,7))}买的公募理财，当时销售起点是多少？请按购买当时有效的规定回答。")
q("VINTAGE", "公募理财", "销售起点", m(2018, 3),
  f"{cn_date(m(2018,3))}买理财，销售起点是多少？")
q("VINTAGE", "个人养老金", "产品范围", m(2023, 3),
  f"我是{cn_date(m(2023,3))}开户的个人养老金参加人，当时账户里的钱可以买哪些产品？请按当时有效的规定回答。")
q("VINTAGE", "个人养老金", "领取条件", m(2023, 6),
  f"{cn_date(m(2023,6))}开户的账户，按当时的规定什么情况下可以领取？")
q("VINTAGE", "个人养老金", "年度缴费上限", m(2023, 3),
  f"{cn_date(m(2023,3))}缴存时，年度上限是多少？")
# ---- STALE: has the standard changed? ----------------------------------------
q("STALE", "大额存单", "起存金额", 150, "听说大额存单门槛降过一次，现在个人起存金额是多少？")
q("STALE", "公募理财", "销售起点", 150, "理财销售起点最近几年是不是调过？现在是多少？")
q("STALE", "个人养老金", "产品范围", 150, "个人养老金可投的产品范围最近是不是扩大了？现在能买哪些？")
q("STALE", "个人养老金", "领取条件", 150, "个人养老金的领取条件最近有没有放宽？现在什么情况能领？")
q("STALE", "信用卡分期", "分期期限上限", 150, "信用卡分期期限的监管上限是多少？最近是不是收紧了？")
q("STALE", "互联网存款", "非自营平台", 150, "第三方平台上的银行存款还能买吗？现在有什么规定？")
# ---- ULTRA: staff/platform claims conflicting with regulation ----------------
q("ULTRA", "公募理财", "保本承诺", 150, "理财经理说这款理财保本保收益、绝对安全，监管允许银行这么说吗？")
q("ULTRA", "公募理财", "预期收益率宣传", 150, "产品海报写着预期年化收益率4.2%，这样的宣传合规吗？")
q("ULTRA", "结构性存款", "收益表述", 150, "客户经理说结构性存款保本保收益稳赚不赔，这符合监管要求吗？")
q("ULTRA", "结构性存款", "投资冷静期", 150, "柜台说现在签约马上生效不用等冷静期，监管是怎么规定的？")
q("ULTRA", "互联网存款", "服务区域", 150, "我在外地，地方银行的APP说全国可存，我能存吗？监管怎么说？")
q("ULTRA", "存款保险", "偿付限额", 150, "网点易拉宝写着存款全额保障存多少赔多少，这跟规定一致吗？")
q("ULTRA", "存款保险", "被保险存款范围", 150, "大堂经理说我买的理财也受存款保险保障，对吗？")
q("ULTRA", "个人养老金", "账户封闭运行", 150, "广告说个人养老金随时可取随用随取，是这样吗？")
q("ULTRA", "公募理财", "代销渠道", 150, "我在第三方平台看到你们行理财可以直接买，这样卖合规吗？")
q("ULTRA", "公募理财", "专区销售", 150, "推广说手机上直接买理财不用录音录像，符合销售规定吗？")
# ---- MARKET: numeric marketing claims vs current rule ------------------------
q("MARKET", "大额存单", "起存金额", 150, "宣传页说大额存单30万起存，现在到底按多少执行？")
q("MARKET", "公募理财", "销售起点", 150, "客服说理财还是5万起购，现在到底多少起买？")
q("MARKET", "个人住房贷款", "日月利率展示", 150, "APP只写日息0.02%不写年化利率，这样合规吗？")
q("MARKET", "信用卡分期", "成本展示形式", 150, "银行把分期成本都叫手续费不叫利息，合规吗？")
q("MARKET", "个人养老金", "产品范围", 110, "直播间说个人养老金想买啥基金就买啥基金，有这种说法的依据吗？")
# ---- SCOPE: transitional grant applies to legacy business only ---------------
q("SCOPE", "信用卡分期", "分期期限上限", 120, "我2023年签的72期账单分期合同，整改要求落到我头上吗？", segment="存量业务")
q("SCOPE", "信用卡分期", "分期期限上限", 120, "存量业务在过渡期内怎么处理分期期限问题？", segment="存量业务")
q("SCOPE", "信用卡分期", "分期期限上限", 130, "过渡期结束后，不符合期限要求的存量分期还能继续吗？", segment="存量业务")
q("SCOPE", "信用卡分期", "分期期限上限", 120, "现在新办分期，期限最长能签到多少年？")
# ---- GAP: no unit in the corpus fixes this value -----------------------------
q("GAP", "大额存单", "提前支取计息规则", 150, "大额存单提前支取按什么利率计息？")
q("GAP", "个人住房贷款", "首套利率下限", 150, "首套房贷利率现在最低能做到多少？")
q("GAP", "个人住房贷款", "提前还款违约金比例", 150, "房贷提前还款要收多少违约金？")
q("GAP", "公募理财", "申赎费率", 150, "开放式理财申购赎回费率是多少？")
q("GAP", "信用卡分期", "最低起始金额", 150, "信用卡账单分期最低起始金额是多少？")
q("GAP", "存款保险", "保费费率", 150, "存款保险的保费费率具体是多少？")
q("GAP", "个人养老金", "领取税负", 150, "领取个人养老金时按百分之几扣税？")


def build_queries(units):
    qs, i = [], 0
    # FACT: every slot with a governing L1/L2 unit at "now"
    slots = sorted({(u.product, u.attr) for u in units
                    if u.tier in OPERATIVE_REAL and u.attr not in
                    ("提前支取计息规则", "首套利率下限", "提前还款违约金比例",
                     "申赎费率", "最低起始金额", "保费费率", "领取税负")})
    for prod, attr in slots:
        ctx = dict(product=prod, attr=attr, region="全部", segment="全部客户", t_ref=NOW_R - 3)
        gold = resolve_real(units, ctx)
        if not gold or not gold[0]:
            continue
        text = Q_OVERRIDES.get((prod, attr),
                               f"请问{prod}的{attr}，现行规定是怎么要求的？")
        gv = gold[0][0]
        qs.append(dict(family="FACT", ctx=ctx, question=text,
                       gold_value=str(gv.value),
                       gold_disp=(f"{gv.value}{gv.unit}" if gv.unit and isinstance(gv.value, (int, float))
                                  else str(gv.value))))
    for spec in Q_EXTRA:
        ctx = dict(product=spec["product"], attr=spec["attr"], region=spec["region"],
                   segment=spec["segment"], t_ref=spec["t_ref"])
        gold = resolve_real(units, ctx)
        gv = gold[0][0] if gold and gold[0] else None
        qs.append(dict(family=spec["family"], ctx=ctx, question=spec["question"],
                       gold_value=str(gv.value) if gv else None,
                       gold_disp=(f"{gv.value}{gv.unit}" if gv and gv.unit and isinstance(gv.value, (int, float))
                                  else (str(gv.value) if gv else None))))
    for k, x in enumerate(qs):
        x["qid"] = f"{x['family']}-{k:03d}"
    return qs


# --------------------------------------------------------------- assembly ----

def assemble_real(units, qd, top_idx, cfg):
    ctx = qd["ctx"]
    cands = [units[i] for i in top_idx]
    if cfg.get("slot"):
        cands = [x for x in cands if x.product == ctx["product"] and x.attr == ctx["attr"]]
    if cfg.get("scope"):
        cands = [x for x in cands if x.region in ("全部", ctx["region"])
                 and x.segment in ("全部客户", ctx["segment"])]
    tref = {"ref": ctx["t_ref"], "now": NOW_R, None: None}[cfg.get("temporal")]
    if tref is not None:
        cands = [x for x in cands if x.valid_from <= tref < x.valid_to]
    if not cands:
        return None, dict(P.EMPTY), []

    if not cfg.get("resolve"):
        op = [x for x in cands if x.tier in OPERATIVE_REAL] or cands
        best = (sorted(op, key=lambda x: (TIER_RANK[x.tier], -x.valid_from))[0]
                if cfg.get("authority") else op[0])
    else:
        gold, cands_op = resolve_real(units, {**ctx, "_cands": cands})
        # resolve_real re-filters by metadata; that equals the filtered view here
        best = gold[0] if gold else None
        if best is None:
            return None, dict(P.EMPTY), cands
    same = [x for x in cands if str(x.value) == str(best.value)]
    other = [x for x in cands if str(x.value) != str(best.value)]
    pos = {id(x): p for p, x in enumerate(cands)}
    fs = qd["_fs"]
    s_best = float(fs[top_idx[pos[id(best)]]]) if id(best) in pos else 0.0
    s_other = max([float(fs[top_idx[pos[id(x)]]]) for x in other], default=0.0)
    tm = min(abs(ctx["t_ref"] - best.valid_from),
             abs(ctx["t_ref"] - best.valid_to) if best.valid_to < 900 else 99)
    f = dict(score=s_best, margin=s_best - s_other,
             uniq=1.0 if (len(same) == 1 and not other) else 0.0,
             agree=len(same) / max(1, len(cands)),
             tmargin=min(tm, 24) / 24.0,
             spec=1.0 if (best.region != "全部" or best.segment != "全部客户") else 0.6,
             n=float(len(cands)))
    cn = cands if not cfg.get("resolve") else [best]
    return best, f, cn


def pick_real(context, qd, memory, eps, rng):
    if not context:
        return None
    # same evidence utility as the benchmark reader, with the real timeline
    def utility(x):
        fs = qd["_fs"]
        try:
            rel = float(fs[id2idx[id(x)]]) / (float(fs[qd["_top"][0]]) + 1e-12)
        except Exception:
            rel = 0.0
        auth = 1.0 / TIER_RANK[x.tier]
        cur = 1.0 if (x.valid_from <= NOW_R < x.valid_to) else 0.25
        sc = 1.0 if (x.region in ("全部", qd["ctx"]["region"])
                     and x.segment in ("全部客户", qd["ctx"]["segment"])) else 0.5
        return 0.45 * rel + 0.30 * auth + 0.15 * cur + 0.10 * sc
    u = max(context, key=utility)
    value = str(u.value)
    if eps > 0:
        alt = [x for x in memory if str(x.value) != value]
        if alt and rng.random() < eps:
            value = str(alt[rng.integers(0, len(alt))].value)
    return value


id2idx = {}


def run_ladder(units, queries, cfg, eps, lam=None, score_fn=None, seed=99):
    rng = np.random.default_rng(seed)
    recs = []
    for qd in queries:
        best, f, context = assemble_real(units, qd, qd["_top"], cfg)
        s = score_fn(f) if score_fn else None
        answered = best is not None and (lam is None or s >= lam)
        val = pick_real(context, qd, [units[i] for i in qd["_top"]], eps, rng) if answered else None
        recs.append(dict(qid=qd["qid"], family=qd["family"], answered=int(answered),
                         value=val, gold=qd["gold_value"],
                         viol=int(answered and val != qd["gold_value"]),
                         s=float(s) if s is not None else None))
    return recs


def summarize(recs):
    n = len(recs)
    ans = [r for r in recs if r["answered"]]
    return dict(n=n, answered=len(ans), coverage=len(ans) / n if n else 0.0,
                bcr=(sum(r["viol"] for r in ans) / len(ans)) if ans else 0.0,
                overall=(sum(r["viol"] for r in recs) / n) if n else 0.0)


LADDER = {
    "M1_vanilla": dict(),
    "M2_slot": dict(slot=True),
    "M3_recency": dict(slot=True, temporal="now"),
    "M4_asof": dict(slot=True, temporal="ref"),
    "M5_scope": dict(slot=True, temporal="ref", scope=True, authority=True),
    "M6_resolve": dict(slot=True, temporal="ref", scope=True, resolve=True),
}


def fit_scorer(units, train_qs, eps=0.05, seed=7):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for qd in train_qs:
        best, f, ctx = assemble_real(units, qd, qd["_top"], LADDER["M6_resolve"])
        if best is None:
            X.append(P.vec(f)); y.append(0); continue
        v = pick_real(ctx, qd, [units[i] for i in qd["_top"]], eps, rng)
        X.append(P.vec(f)); y.append(int(v == qd["gold_value"]))
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    return lambda f: float(lr.predict_proba(sc.transform([P.vec(f)]))[0, 1])


def main():
    units = build_real_kb()
    qs = build_queries(units)
    texts = [u.text for u in units]
    bm, dn = P.BM25(texts), P.Dense(texts)
    for qd in qs:
        f = P.rrf(bm.scores(qd["question"]), dn.scores(qd["question"]))
        qd["_top"], qd["_fs"] = list(np.argsort(-f)[:K]), f
    global id2idx
    id2idx = {id(x): i for i, x in enumerate(units)}

    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(qs))
    tr = [qs[i] for i in idx[: len(qs) // 2]]
    te = [qs[i] for i in idx[len(qs) // 2:]]
    fit = fit_scorer(units, tr)

    res = {"meta": dict(units=len(units), queries=len(qs), train=len(tr), test=len(te),
                        now=NOW_R,
                        families={f: sum(1 for x in qs if x["family"] == f)
                                  for f in ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]},
                        tiers={t: sum(1 for x in units if x.tier == t)
                               for t in ["L1", "L2", "L5"]},
                        doc_units=sum(1 for x in units if x.doc_id not in ()),
                        ),
           "ladder": {}, "control": {}}
    for name, cfg in LADDER.items():
        recs = run_ladder(units, te, cfg, 0.05)
        s = summarize(recs)
        s["by_family"] = {f: summarize([r for r in recs if r["family"] == f])
                          for f in ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]}
        res["ladder"][name] = s

    n = len(te)
    cal, ev = te[: n // 2], te[n // 2:]
    best_lam, best_cov = 0.995, -1
    for lam in np.linspace(0.02, 0.99, 80):
        s = summarize(run_ladder(units, cal, LADDER["M6_resolve"], 0.05,
                                 lam=float(lam), score_fn=fit))
        if s["bcr"] <= 0.05 and s["coverage"] > best_cov:
            best_lam, best_cov = float(lam), s["coverage"]
    recs = run_ladder(units, ev, LADDER["M6_resolve"], 0.05, lam=best_lam, score_fn=fit)
    s = summarize(recs); s["lambda"] = best_lam
    res["control"]["static"] = s
    lam, recs = 0.5, []
    for qd in ev:
        r = run_ladder(units, [qd], LADDER["M6_resolve"], 0.05, lam=lam, score_fn=fit)[0]
        recs.append(r)
        lam = min(0.999, max(0.001, lam + 0.10 * (r["viol"] - 0.05)))
    res["control"]["aci"] = summarize(recs)

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3500])


if __name__ == "__main__":
    main()
