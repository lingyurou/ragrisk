# -*- coding: utf-8 -*-
"""Hand-rolled SVG figures (no plotting dependency) from results.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

W, H = 520, 330
PAD_L, PAD_R, PAD_T, PAD_B = 58, 16, 16, 46
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
C = {"M1_vanilla": "#9ca3af", "M4_asof": "#f59e0b", "M5_scope": "#a855f7",
     "M6_resolve": "#2563eb", "M7_aci": "#dc2626", "static": "#059669"}


def sx(x, xmax):
    return PAD_L + x / xmax * (W - PAD_L - PAD_R)


def sy(y, ymax):
    return H - PAD_B - y / ymax * (H - PAD_T - PAD_B)


def path(xs, ys, xmax, ymax, color, width=2.0, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x, xmax):.1f},{sy(y, ymax):.1f}"
                 for i, (x, y) in enumerate(zip(xs, ys)))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
            f' stroke-linejoin="round" stroke-linecap="round"{da}/>')


def axes(xmax, ymax, xlabel, ylabel, xticks, yticks, fmt_x=lambda v: f"{v:.1f}",
         fmt_y=lambda v: f"{v:.2f}"):
    s = []
    for t in yticks:
        y = sy(t, ymax)
        s.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{PAD_L - 8}" y="{y + 3.5:.1f}" font-size="10" fill="{MUTED}" '
                 f'text-anchor="end">{fmt_y(t)}</text>')
    for t in xticks:
        x = sx(t, xmax)
        s.append(f'<line x1="{x:.1f}" y1="{H - PAD_B}" x2="{x:.1f}" y2="{H - PAD_B + 4}" stroke="{MUTED}"/>')
        s.append(f'<text x="{x:.1f}" y="{H - PAD_B + 17}" font-size="10" fill="{MUTED}" '
                 f'text-anchor="middle">{fmt_x(t)}</text>')
    s.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" stroke="{MUTED}"/>')
    s.append(f'<text x="{(PAD_L + W - PAD_R) / 2}" y="{H - 8}" font-size="11" fill="{INK}" '
             f'text-anchor="middle">{xlabel}</text>')
    s.append(f'<text x="14" y="{(PAD_T + H - PAD_B) / 2}" font-size="11" fill="{INK}" '
             f'text-anchor="middle" transform="rotate(-90 14 {(PAD_T + H - PAD_B) / 2})">{ylabel}</text>')
    return "\n".join(s)


def svg(body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>{body}</svg>')


# ---------------------------------------------------------------- Fig 2: risk-coverage
def fig_risk_coverage(key="e0.05"):
    rc = R["risk_coverage"]
    body = [axes(1.0, 1.0, "coverage", "binding-inconsistency risk",
                 [0, .2, .4, .6, .8, 1.0], [0, .2, .4, .6, .8, 1.0])]
    labels = {"M1_vanilla": "M1 vanilla RAG", "M4_asof": "M4 +as-of",
              "M5_scope": "M5 +scope/authority", "M6_resolve": "M6 BIND-RAG"}
    for k in ["M1_vanilla", "M4_asof", "M5_scope", "M6_resolve"]:
        d = rc.get(k)
        if not d:
            continue
        body.append(path(d["coverage"], d["risk"], 1.0, 1.0, C[k], 2.2))
    # legend
    y0 = PAD_T + 6
    for i, k in enumerate(["M1_vanilla", "M4_asof", "M5_scope", "M6_resolve"]):
        yy = y0 + i * 15
        body.append(f'<line x1="{W - 190}" y1="{yy}" x2="{W - 168}" y2="{yy}" stroke="{C[k]}" stroke-width="2.4"/>')
        body.append(f'<text x="{W - 162}" y="{yy + 3.5}" font-size="10" fill="{INK}">{labels[k]}</text>')
    return svg("\n".join(body))


# ---------------------------------------------------------------- Fig 3: ACI tracking
def fig_aci(key="e0.05"):
    d = R["grid"][key]["M7_aci"]
    curve = d["risk_curve"]
    n = len(curve)
    body = [axes(n, 0.30, "query index in the replayed stream", "cumulative violation rate",
                 [0, 50, 100, 150, n], [0, .05, .10, .15, .20, .25, .30],
                 fmt_x=lambda v: f"{int(v)}", fmt_y=lambda v: f"{v:.2f}")]
    body.append(path([0, n], [0.05, 0.05], n, 0.30, "#059669", 1.6, dash="6 4"))
    body.append(f'<text x="{sx(n, n) - 6}" y="{sy(0.05, 0.30) - 6}" font-size="10" '
                f'fill="#059669" text-anchor="end">target alpha = 0.05</text>')
    body.append(path(list(range(1, n + 1)), curve, n, 0.30, C["M7_aci"], 2.2))
    body.append(f'<text x="{PAD_L + 8}" y="{PAD_T + 12}" font-size="10" fill="{INK}">'
                f'adaptive conformal inference (gamma = 0.1)</text>')
    return svg("\n".join(body))


# ---------------------------------------------------------------- Fig 4: family bars
def fig_family(key="e0.05"):
    fam = ["FACT", "STALE", "VINTAGE", "ULTRA", "MARKET", "SCOPE", "GAP"]
    ms = ["M1_vanilla", "M6_resolve", "M7_aci"]
    w, gap = 22, 6
    body = [axes(len(fam), 1.0, "", "violation rate by question family",
                 [], [0, .2, .4, .6, .8, 1.0])]
    for i, f in enumerate(fam):
        x0 = PAD_L + 18 + i * ((W - PAD_L - PAD_R - 30) / len(fam))
        for j, m in enumerate(ms):
            v = R["grid"][key][m]["by_family"][f]["bcr"]
            h = (1 - v) * 0  # placeholder
            y = sy(v, 1.0)
            body.append(f'<rect x="{x0 + j * (w + gap):.1f}" y="{y:.1f}" width="{w}" '
                        f'height="{H - PAD_B - y:.1f}" fill="{C[m]}" opacity="0.9"/>')
        body.append(f'<text x="{x0 + (3 * w + 2 * gap) / 2:.1f}" y="{H - PAD_B + 16}" font-size="9.5" '
                    f'fill="{MUTED}" text-anchor="middle">{f}</text>')
    for i, m in enumerate(ms):
        yy = PAD_T + 4 + i * 14
        body.append(f'<rect x="{W - 168}" y="{yy - 6}" width="10" height="10" fill="{C[m]}"/>')
        body.append(f'<text x="{W - 154}" y="{yy + 3}" font-size="9.5" fill="{INK}">'
                    f'{{"M1_vanilla": "M1 vanilla", "M6_resolve": "M6 BIND-RAG", "M7_aci": "M6 + ACI"}}[m]'
                    .replace('{"M1_vanilla": "M1 vanilla", "M6_resolve": "M6 BIND-RAG", "M7_aci": "M6 + ACI"}[m]',
                             {"M1_vanilla": "M1 vanilla", "M6_resolve": "M6 BIND-RAG", "M7_aci": "M6 + ACI"}[m])
                    + '</text>')
    return svg("\n".join(body))


# ---------------------------------------------------------------- Fig 5: rho robustness
def fig_rho():
    rhos = [0.0, 0.10, 0.15, 0.25]
    keys = ["r0.0", "r0.1", None, "r0.25"]
    ms = ["M1_vanilla", "M6_resolve", "M7_aci"]
    body = [axes(0.30, 1.0, "metadata corruption rate (rho)", "violation rate among answered",
                 [0, .1, .15, .2, .25], [0, .2, .4, .6, .8, 1.0])]
    for m in ms:
        xs, ys = [], []
        for r, k in zip(rhos, keys):
            if k is None:
                v = R["grid"]["e0.05"][m]["bcr"] if m in R["grid"]["e0.05"] else 0
            else:
                v = R["rho"][k][m]["bcr"]
            xs.append(r); ys.append(v)
        body.append(path(xs, ys, 0.30, 1.0, C[m], 2.2))
        for x, y in zip(xs, ys):
            body.append(f'<circle cx="{sx(x, 0.30):.1f}" cy="{sy(y, 1.0):.1f}" r="2.8" fill="{C[m]}"/>')
    names = {"M1_vanilla": "M1 vanilla", "M6_resolve": "M6 BIND-RAG", "M7_aci": "M6 + ACI"}
    for i, m in enumerate(ms):
        yy = PAD_T + 4 + i * 14
        body.append(f'<line x1="{W - 170}" y1="{yy}" x2="{W - 148}" y2="{yy}" stroke="{C[m]}" stroke-width="2.4"/>')
        body.append(f'<text x="{W - 142}" y="{yy + 3.5}" font-size="9.5" fill="{INK}">{names[m]}</text>')
    return svg("\n".join(body))


for name, fn in [("fig2_risk_coverage", fig_risk_coverage), ("fig3_aci_tracking", fig_aci),
                 ("fig4_family", fig_family), ("fig5_rho", fig_rho)]:
    open(f"{OUT}/{name}.svg", "w").write(fn())
    print("wrote", name)
