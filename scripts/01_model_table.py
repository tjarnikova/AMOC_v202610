#!/usr/bin/env python
"""Summary table of the TOM12_TJ AMOC hosing runs (old runset HA01/HH01 + 8 new runs).

Page-sized (5 x 3 in). Writes plts/model_table.{png,pdf,csv}.
Font: Jost (OFL) from ../fonts; falls back to DejaVu Sans if not found.
Run from anywhere:  python scripts/01_model_table.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plts"; OUT.mkdir(exist_ok=True)

# ---- font ---------------------------------------------------------------------
FONT = "DejaVu Sans"
fdir = ROOT / "fonts"
for f in ["Jost-Regular.ttf", "Jost-Medium.ttf", "Jost-SemiBold.ttf"]:
    if (fdir / f).exists():
        fm.fontManager.addfont(str(fdir / f)); FONT = "Jost"
if FONT != "Jost":
    print("WARNING: Jost not found in", fdir, "- using DejaVu Sans")

# ---- style --------------------------------------------------------------------
colors = {'G2': '#3A5B8C', 'G3': '#6B9BD8', 'R2': '#B23D32', 'R3': '#F06C1B', 'old': 'black'}
linestyle = {'A': '-', 'H': ':'}
INK, GREY, BAND = '#222222', '#8A8A8A', '#F3F1EE'

# ---- specs --------------------------------------------------------------------
# (group label, [pair, ...]); pair = dict shared by its A and H run.
HOL = {'2': 'bgc2 · Holling II', '3': 'bgc3 · Holling III'}
SRC = {'2': 'TOM12_RY_JRFD',     '3': 'TOM12_RY_JRFE'}

def new_pair(key):
    rs, b = key[0], key[1]
    spin = 'FRA8' if rs == 'R' else 'FGA8'
    return dict(key=key, bgc=(HOL[b], SRC[b]),
                runs=[dict(name=f'TOM12_TJ_{key}A2', ah='A', hosing='off',    period='1920–2024',  start=f'{spin} restart'),
                      dict(name=f'TOM12_TJ_{key}H2', ah='H', hosing='0.3 Sv', period='1940–2024*', start=f'{key}A2, 1940')])

groups = [
    ("Original runset  ·  no salinity restoring",
     [dict(key='old', bgc=('Obio', 'close to bgc2'),
           runs=[dict(name='TOM12_TJ_HA01', ah='A', hosing='off', period='1940–2024', start='end of SM00'),
                 dict(name='TOM12_TJ_HH01', ah='H', hosing='on',  period='1940–2024', start='end of SM00')])]),
    ("Salinity restoring + runoff  (R)", [new_pair('R2'), new_pair('R3')]),
    ("No salinity restoring  (G)",       [new_pair('G2'), new_pair('G3')]),
]

# csv
recs = []
for g, pairs in groups:
    for p in pairs:
        for r in p['runs']:
            recs.append(dict(group=g, run=r['name'], bgc=' / '.join(p['bgc']), hosing=r['hosing'],
                             period=r['period'], start_from=r['start']))
pd.DataFrame(recs).to_csv(OUT / "model_table.csv", index=False)

# ---- draw ---------------------------------------------------------------------
sns.set_theme(style="white", rc={'font.family': FONT})
W, H = 5.0, 3.0
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.axis('off')
ax.set_xlim(0, 1)

heads = ['Run', '', 'BGC', 'Hosing', 'Period', 'Start from']
xs    = [0.00, 0.27, 0.40, 0.64, 0.75, 0.865]
FS = 6.4
nunits = 1 + sum(1 + 2 * len(p) for _, p in groups)       # header + group rows + run rows
foot_units = 2.6
ax.set_ylim(-(nunits + foot_units), 0.5)

y = 0
for x, h in zip(xs, heads):
    ax.text(x, y, h, fontsize=FS + 0.6, fontweight='semibold', va='center', color=INK)
ax.plot([0, 1], [y - 0.62, y - 0.62], color=INK, lw=0.9)
y -= 1

for gi, (glabel, pairs) in enumerate(groups):
    y -= 0.15
    ax.text(0, y, glabel, fontsize=FS + 0.2, fontweight='semibold', va='center', color=colors[pairs[0]['key']] if pairs[0]['key'] != 'old' else INK)
    ax.plot([0, 1], [y - 0.5, y - 0.5], color='#BBBBBB', lw=0.5)
    y -= 1
    for pi, p in enumerate(pairs):
        ytop = y
        if pi % 2 == 0:
            ax.fill_between([0, 1], ytop + 0.5, ytop - 1.5, color=BAND, zorder=0, lw=0)
        for r in p['runs']:
            ax.text(xs[0], y, r['name'], fontsize=FS, va='center', color=INK)
            ax.plot([xs[1], xs[1] + 0.075], [y, y], color=colors[p['key']], ls=linestyle[r['ah']],
                    lw=1.9, solid_capstyle='butt', dash_capstyle='butt')
            ax.text(xs[3], y, r['hosing'], fontsize=FS, va='center', color=INK)
            ax.text(xs[4], y, r['period'], fontsize=FS, va='center', color=INK)
            ax.text(xs[5], y, r['start'], fontsize=FS, va='center', color=INK)
            y -= 1
        ymid = ytop - 0.5                                      # merged BGC cell over the A/H pair
        ax.text(xs[2], ymid + 0.27, p['bgc'][0], fontsize=FS, va='center', color=INK)
        ax.text(xs[2], ymid - 0.30, p['bgc'][1], fontsize=FS - 0.9, va='center', color=GREY)

ax.plot([0, 1], [y + 0.5, y + 0.5], color=INK, lw=0.9)
note = ("A = no hosing; H = 0.3 Sv hosing branching from 1940 of the matching A run.  "
        "Dotted = H, solid = A.\n* H end year assumed (unconfirmed).  "
        "bgc2 / bgc3 = Holling type II / III.")
ax.text(0, y - 0.35, note, fontsize=FS - 1.2, va='top', color=GREY, linespacing=1.35)

fig.savefig(OUT / "model_table.png", dpi=300)
fig.savefig(OUT / "model_table.pdf")
print("wrote", OUT, "| font:", FONT)
