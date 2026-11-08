# -*- coding: utf-8 -*-
"""
BankKB-C: a semi-synthetic benchmark for binding-grounded answering over a
customer-facing banking knowledge base (bitemporal + authority-stratified).

Deterministic given SEED. Gold labels are constructed by the generator, so the
binding set is exact by construction rather than annotated.

Normative resolution (paper, Def. 3 / Alg. 1):
  R1 admissibility : temporally valid at the reference time, scope-matching, and
                     not laxer than any higher-authority applicable norm --
                     unless the unit is an explicitly delegated grant.
  R2 specificity   : lex specialis -- the most specific admissible unit governs.
  R3 severity      : lex stringentior -- ties resolved toward the stricter norm.
  R4 authority     : higher tier wins remaining ties.
  R5 recency       : later enactment wins the last tie.
"""
import random, itertools
from dataclasses import dataclass

SEED = 20261029

REGIONS = ["华东", "华南", "华北", "西部"]
SEGMENTS = ["全部客户", "代发工资客户", "高净值客户", "小微企业主"]
TIERS = {"L1": ("监管规定", 1), "L2": ("总行管理办法", 2), "L3": ("分行实施细则", 3),
         "L4": ("产品说明书", 4), "L5": ("营销宣传与FAQ", 5)}
TIER_RANK = {k: v[1] for k, v in TIERS.items()}

PRODUCTS = [
    ("个人住房贷款", {"单户额度上限": (600, "万元", [600, 500, 500, 450]),
                 "年化利率": (3.45, "%", [3.45, 3.20, 3.05, 2.85]),
                 "最长贷款期限": (30, "年", [30, 30, 25, 25]),
                 "最低首付比例": (20, "%", [20, 20, 15, 15])}),
    ("个人消费贷款", {"单户额度上限": (50, "万元", [50, 40, 30, 30]),
                 "年化利率": (3.60, "%", [3.60, 3.40, 3.40, 3.10]),
                 "最长贷款期限": (5, "年", [5, 5, 3, 3])}),
    ("个人经营性贷款", {"单户额度上限": (500, "万元", [500, 400, 400, 300]),
                  "年化利率": (3.85, "%", [3.85, 3.65, 3.45, 3.45]),
                  "最长贷款期限": (10, "年", [10, 8, 8, 5])}),
    ("信用卡分期", {"单期手续费率": (0.75, "%", [0.75, 0.72, 0.66, 0.60]),
                "单户分期额度上限": (30, "万元", [30, 30, 20, 20]),
                "最长期数": (36, "期", [36, 36, 24, 24])}),
    ("大额存单", {"起存金额": (20, "万元", [20, 20, 30, 30]),
              "三年期利率": (2.15, "%", [2.15, 2.60, 2.35, 2.10]),
              "可转让性": ("可转让", "", ["可转让", "可转让", "不可转让", "不可转让"])}),
    ("个人养老金账户", {"年度缴存上限": (1.2, "万元", [1.2] * 4),
                  "可投资产品范围": ("储蓄、理财、基金、保险", "", ["储蓄、理财、基金、保险"] * 4)}),
]

T_L2 = ("{product}管理办法（{ver}）第{art}条：本行{product}{attr}执行标准为{v}{unit}，适用于全行各分支机构。"
        "本条款自{date}起施行，原相关规定同时废止。")
T_L3 = ("{region}分行关于{product}的实施细则（{ver}）第{art}条：结合区域市场与风险管理要求，"
        "本分行辖内{product}{attr}按{v}{unit}执行，自{date}起施行。")
T_L4 = ("{product}产品说明书（{ver}）第{art}条：客户签约时适用的{attr}为{v}{unit}，"
        "具体以合同约定及签约日有效的本行规定为准。")
T_L5 = ("【常见问题】{product}{attr}是多少？目前{product}{attr}为{v}{unit}，详情请咨询网点或拨打客服热线。")
T_L1 = ("关于加强{product}业务管理的通知（{ver}）第{art}条：商业银行开展{product}业务，{attr}应满足"
        "{v}{unit}的审慎要求，不得突破。")
FAQS_EXTRA = [
    "客户如何查询{product}的还款计划？可通过手机银行“贷款-我的贷款”查看。",
    "申请{product}需要准备哪些材料？一般包括身份证明、收入证明与用途说明。",
    "{product}能否提前还款？提前还款需提前{n}个工作日提出申请。",
    "逾期后如何处理？请及时联系客户经理，逾期记录将按规定报送征信系统。",
    "{product}的审批时限是多久？资料齐全情况下一般为{n}个工作日。",
    "外籍客户是否可以申请{product}？需满足监管关于非居民客户的准入要求。",
    "{product}支持哪些还款方式？支持等额本息、等额本金与按月付息到期还本。",
    "存量客户如何办理{product}的展期？需在到期前{n}个工作日提交申请。",
]


def cn_date(m):
    return f"{2023 + m // 12}年{m % 12 + 1}月1日"


@dataclass
class Unit:
    uid: str
    doc_id: str
    tier: str
    product: str
    attr: str
    value: object
    unit: str
    region: str
    segment: str
    valid_from: int
    valid_to: int
    tx_version: int
    text: str
    ver: str
    grant: int = 0          # 1 = explicitly delegated exception (may be laxer within scope)


def build_kb(seed=SEED, T=36):
    rng = random.Random(seed)
    units, uid = [], itertools.count(1)
    for prod, attrs in PRODUCTS:
        revs = sorted(rng.sample(range(4, T - 4), 3))
        bounds = [0] + revs + [T]
        for pi, (attr, (base, unit, series)) in enumerate(attrs.items()):
            expired = (pi % 3 == 0)      # last known version lapses without a successor
            for k in range(len(bounds) - 1):
                vf, vt = bounds[k], bounds[k + 1]
                val = series[min(k, len(series) - 1)]
                is_last = (k == len(bounds) - 2)
                end = (T if (expired and is_last) else (vt if vt < T else 999))
                units.append(Unit(f"u{next(uid)}", f"DOC-L2-{prod}-{attr}-{k}", "L2", prod, attr,
                                  val, unit, "全部", "全部客户", vf, end, k,
                                  T_L2.format(product=prod, attr=attr, v=val, unit=unit,
                                              ver=f"{chr(65 + k)}版", art=10 + k, date=cn_date(vf)),
                                  f"{chr(65 + k)}版"))
            if isinstance(base, (int, float)) and (attr.endswith("额度上限") or attr == "起存金额"
                                                   or attr == "最低首付比例"):
                ceiling = round(base * 1.1, 2) if attr.endswith("额度上限") else (
                    base * 0.85 if attr == "最低首付比例" else round(base * 0.85, 2))
                units.append(Unit(f"u{next(uid)}", f"DOC-L1-{prod}-{attr}", "L1", prod, attr,
                                  ceiling, unit, "全部", "全部客户", 0, 999, 0,
                                  T_L1.format(product=prod, attr=attr, v=ceiling, unit=unit,
                                              ver="2024", art=8), "2024"))
            for region in rng.sample(REGIONS, 2):
                if not isinstance(base, (int, float)) or unit == "":
                    continue
                vf = rng.choice([0, 6, 12, 18])
                vt = 999 if rng.random() < 0.5 else vf + rng.choice([12, 18])
                lax = rng.random() < 0.35
                if attr.endswith("额度上限"):
                    bval = round(base * (1.25 if lax else 0.8), 2)
                elif attr == "最低首付比例":
                    bval = base - 5 if lax else base + 5
                elif attr == "起存金额":
                    bval = round(base * 0.75 if lax else base * 1.5, 2)
                else:
                    bval = round(base * (1.1 if lax else 0.9), 2)
                units.append(Unit(f"u{next(uid)}", f"DOC-L3-{prod}-{attr}-{region}", "L3", prod,
                                  attr, bval, unit, region, "全部客户", vf, vt, 0,
                                  T_L3.format(product=prod, attr=attr, v=bval, unit=unit,
                                              region=region, ver="2025修订", art=5, date=cn_date(vf)),
                                  "2025修订"))
            units.append(Unit(f"u{next(uid)}", f"DOC-L4-{prod}-{attr}", "L4", prod, attr, base, unit,
                              "全部", "全部客户", 0, 999, 0,
                              T_L4.format(product=prod, attr=attr, v=base, unit=unit, ver="首版", art=2),
                              "首版"))
            lag = series[max(0, len(series) - 3)]
            if isinstance(lag, (int, float)):
                lag = round(lag * (1.05 if rng.random() < 0.5 else 1.0), 2)
            units.append(Unit(f"u{next(uid)}", f"DOC-L5-{prod}-{attr}", "L5", prod, attr, lag, unit,
                              "全部", "全部客户", 0, 999, 0,
                              T_L5.format(product=prod, attr=attr, v=lag, unit=unit), "宣传物料"))

    for i in range(620):
        prod = rng.choice([p for p, _ in PRODUCTS])
        txt = rng.choice(FAQS_EXTRA).format(product=prod, n=rng.choice([3, 5, 10]))
        units.append(Unit(f"u{next(uid)}", f"DOC-FAQ-{i}", "L5", prod, "流程", "", "", "全部",
                          "全部客户", 0, 999, 0, txt, "FAQ"))

    for prod, cap, seg in [("个人消费贷款", 60, "代发工资客户"),
                           ("个人经营性贷款", 600, "小微企业主"),
                           ("个人住房贷款", 700, "高净值客户")]:
        vf = rng.choice([6, 12])
        units.append(Unit(f"u{next(uid)}", f"DOC-SEG-{prod}", "L3", prod, "单户额度上限", cap,
                          "万元", "华东", seg, vf, 999, 0,
                          f"{prod}优质客群专项方案（经总行授权）：面向{seg}，{prod}单户额度上限可上浮至"
                          f"{cap}万元，其他客群不适用。自{cn_date(vf)}起执行。", "专项方案", grant=1))
    return units, T


def scope_specificity(u):
    return (u.region != "全部") + (u.segment != "全部客户")


def applicable(u, ctx):
    return (u.product == ctx["product"] and u.attr == ctx["attr"]
            and u.region in ("全部", ctx["region"])
            and u.segment in ("全部客户", ctx["segment"])
            and u.valid_from <= ctx["t_ref"] < u.valid_to)


def _num(u):
    try:
        return float(u.value)
    except (TypeError, ValueError):
        return None


def is_laxer(u, v):
    """True if u permits a laxer outcome than v; None when incomparable (text)."""
    a, b = _num(u), _num(v)
    if a is None or b is None:
        return None
    if u.attr in ("起存金额", "最低首付比例"):
        return a < b
    return a > b


OPERATIVE = ("L2", "L3", "L4")


def resolve(units, ctx):
    cands = [u for u in units if applicable(u, ctx)]
    if not cands:
        return [], []
    cands_op = [u for u in cands if u.tier in OPERATIVE]
    if not cands_op:
        return [], cands
    cands = cands_op
    adm = []
    for u in cands:
        ok = True
        if not u.grant:
            for v in cands:
                if TIER_RANK[v.tier] < TIER_RANK[u.tier] and is_laxer(u, v) is True:
                    ok = False
                    break
        if ok:
            adm.append(u)
    if not adm:
        return [], cands
    # R2 specificity -> R4 authority -> R3 severity -> R5 recency
    max_spec = max(scope_specificity(u) for u in adm)
    pool = [u for u in adm if scope_specificity(u) == max_spec]
    top = min(TIER_RANK[u.tier] for u in pool)
    pool = [u for u in pool if TIER_RANK[u.tier] == top]
    pool.sort(key=lambda u: -u.valid_from)
    best = pool[0]
    for u in pool:                       # R3: stricter prevails within the same tier
        if is_laxer(best, u) is True:
            best = u
    return [best], cands


FAMILIES = ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]

# product x attribute pairs that the knowledge base simply does not cover
GAP_PAIRS = [("个人消费贷款", "提前还款违约金比例"), ("个人住房贷款", "组合贷商贷部分利率"),
             ("大额存单", "提前支取计息规则"), ("信用卡分期", "提前结清手续费"),
             ("个人经营性贷款", "随借随还额度恢复规则"), ("个人养老金账户", "领取时计税方式")]


N_PER_FAMILY = {"FACT": 360, "STALE": 66, "VINTAGE": 66, "ULTRA": 66, "MARKET": 66, "SCOPE": 66, "GAP": 66}


def build_queries(units, T, seed=SEED, n_per_family=None):
    n_per_family = n_per_family or N_PER_FAMILY
    rng = random.Random(seed + 7)
    keys = sorted({(u.product, u.attr) for u in units if u.attr and u.unit != ""})
    buckets = {f: [] for f in FAMILIES}
    guard = 0
    while any(len(buckets[f]) < n_per_family[f] for f in FAMILIES) and guard < 400000:
        guard += 1
        prod, attr = rng.choice(keys)
        ctx = {"product": prod, "attr": attr, "region": rng.choice(REGIONS + ["全部", "全部"]),
               "segment": rng.choice(SEGMENTS + ["全部客户"]), "t_ref": rng.randint(8, T - 1)}
        gold = resolve(units, ctx)[0]
        fam = None
        if not gold:
            continue
        g = gold[0]

        # VINTAGE: the customer signed before the operative clause took effect
        if g.valid_from > 0 and rng.random() < 0.4:
            tr = rng.randint(max(0, g.valid_from - 10), g.valid_from - 1)
            ctx_v = dict(ctx, t_ref=tr)
            gv = resolve(units, ctx_v)[0]
            if gv and str(gv[0].value) != str(g.value):
                fam, ctx = "VINTAGE", ctx_v

        # ULTRA: an applicable branch rule is laxer and not a delegated grant
        if fam is None and rng.random() < 0.4:
            lax = [u for u in units if applicable(u, ctx) and u.tier == "L3" and not u.grant
                   and is_laxer(u, g) is True]
            if lax:
                fam = "ULTRA"
                ctx = dict(ctx, region=lax[0].region, segment="全部客户")
                if not resolve(units, ctx)[0]:
                    fam = None

        # SCOPE: the customer is outside the segment the grant is issued to
        if fam is None and rng.random() < 0.35:
            gr = [u for u in units if u.grant and u.product == prod and u.attr == attr]
            if gr:
                s = gr[0]
                fam = "SCOPE"
                ctx = dict(ctx, region=s.region, segment=rng.choice([x for x in SEGMENTS if x != s.segment]))

        # MARKET: a marketing unit asserts a different number
        if fam is None and rng.random() < 0.45:
            mkt = [u for u in units if u.product == prod and u.attr == attr and u.tier == "L5"
                   and u.unit != "" and str(u.value) != str(g.value)]
            if mkt:
                fam = "MARKET"

        # STALE: the operative clause was revised at least once
        if fam is None and g.valid_from > 0 and rng.random() < 0.5:
            fam = "STALE"

        # GAP: no applicable unit in the query window
        if fam is None and rng.random() < 0.15:
            ctx_g = dict(ctx, t_ref=T + rng.randint(1, 6))
            if not resolve(units, ctx_g)[0]:
                fam, ctx = "GAP", ctx_g

        if fam is None:
            fam = "FACT"
        if len(buckets[fam]) >= n_per_family[fam]:
            continue
        gold = resolve(units, ctx)[0]
        if not gold:
            continue
        buckets[fam].append({"family": fam, "ctx": ctx, "question": make_question(ctx, fam),
                             "gold_value": str(gold[0].value), "gold_units": [u.uid for u in gold],
                             "gold_tier": gold[0].tier})
    out = [x for v in buckets.values() for x in v]
    # GAP family: attributes the knowledge base never covers
    while len(buckets["GAP"]) < n_per_family["GAP"]:
        prod, attr = rng.choice(GAP_PAIRS)
        ctx = {"product": prod, "attr": attr, "region": rng.choice(REGIONS),
               "segment": rng.choice(SEGMENTS), "t_ref": rng.randint(8, T - 1)}
        assert not resolve(units, ctx)[0]
        buckets["GAP"].append({"family": "GAP", "ctx": ctx, "question": make_question(ctx, "GAP"),
                               "gold_value": None, "gold_units": [], "gold_tier": None})
    out = [x for v in buckets.values() for x in v]
    rng.shuffle(out)
    for i, x in enumerate(out):
        x["idx"], x["qid"] = i, f"{x['family']}-{i:03d}"
    return out


def make_question(ctx, fam):
    p, a = ctx["product"], ctx["attr"]
    reg = "" if ctx["region"] == "全部" else f"在{ctx['region']}"
    if fam == "VINTAGE":
        return f"我是在{cn_date(ctx['t_ref'])}签约的{p}客户，当时适用的{a}是多少？请按签约时有效的本行规定回答。"
    if fam == "GAP":
        return f"请问{p}的{a}目前执行什么标准？"
    if fam == "MARKET":
        return f"宣传页上和网点说的{p}{a}不一致，到底按哪个执行？"
    if fam == "SCOPE":
        return f"我是{ctx['segment']}，{reg}办理{p}，{a}按什么标准？"
    if fam == "ULTRA":
        return f"我是{ctx['segment']}，{reg}办理{p}，分行细则和总行规定不一样，{a}到底按哪个执行？"
    if fam == "STALE":
        return f"{p}的{a}最近调整过吗？现在执行的标准是多少？"
    return f"{p}的{a}是多少？"


if __name__ == "__main__":
    from collections import Counter
    units, T = build_kb()
    qs = build_queries(units, T)
    print("units", len(units), "queries", len(qs), Counter(q["family"] for q in qs))
    print("no-gold", sum(1 for q in qs if not q["gold_units"]))
