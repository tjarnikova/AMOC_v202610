#!/usr/bin/env python
"""
B3_bars.py  --  bar-plot alternative to B2_polar_biomass.py.

3 stacked panels (NHL, NA, AB); x = PFT, y = regional depth-integrated biomass (PgC, FULL COLUMN).
Per PFT: one filled bar per run set (no-hosing A run); the hosing (H) run is an open marker of the
same colour at the same x. Value = mean(<name>_int) x region_area_m2 x 12.011e-15, averaged over
a window of annual means (same reader and assumptions as B2).

Input : <CLIMS>/<model>/timeseries/<model>_ptrc_T_<YYYY>_regmean_weekly.nc
Output: <PLTS>/B3_bars_<set>_<y0>-<y1>.{png,pdf}   set = all | oldG

Env overrides: CLIMS, PLTS, FONTS, WINDOW (default 2015-2019),
  LOGY=1  log y-axis (shows the small PFTs; default linear)
"""
import os
import re
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CLIMS = os.environ.get("CLIMS", "/gpfs/scratch/greenocean/mep22dku/clims")
PLTS = os.environ.get("PLTS", os.path.join(HERE, "..", "plts"))
FONTS = os.environ.get("FONTS", os.path.join(HERE, "..", "fonts"))
WINDOW = os.environ.get("WINDOW", "2015-2019")
LOGY = os.environ.get("LOGY", "0") == "1"
Y0, Y1 = [int(v) for v in WINDOW.split("-")]
FT = "ptrc_T"
os.makedirs(PLTS, exist_ok=True)

# ---------------------------------------------------------------- style
for f in glob.glob(os.path.join(FONTS, "*.ttf")) + glob.glob(os.path.join(FONTS, "*.otf")):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
try:
    import seaborn as sns
    sns.set_theme(style="ticks")
except Exception:
    print("seaborn not found, using plain matplotlib")
_have = {f.name for f in font_manager.fontManager.ttflist}
if "Jost" in _have:
    plt.rcParams["font.family"] = ["Jost", "DejaVu Sans"]
else:
    print("Jost not found in %s, using DejaVu Sans" % FONTS)
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 6

colors = {"G2": "#0072B2", "G3": "#56B4E9", "R2": "#B23D32", "R3": "#F06C1B", "old": "black"}
linestyle = {"A": "-", "H": ":"}
MODELS = {
    "TOM12_TJ_HA01": ("old", "A"), "TOM12_TJ_HH01": ("old", "H"),
    "TOM12_TJ_G2A2": ("G2", "A"), "TOM12_TJ_G2H2": ("G2", "H"),
    "TOM12_TJ_G3A2": ("G3", "A"), "TOM12_TJ_G3H2": ("G3", "H"),
    "TOM12_TJ_R2A2": ("R2", "A"), "TOM12_TJ_R2H2": ("R2", "H"),
    "TOM12_TJ_R3A2": ("R3", "A"), "TOM12_TJ_R3H2": ("R3", "H"),
}
LABELS = {"old": "Old runset (HA01/HH01)", "G2": "G2 (no restoring, Holling II)",
          "G3": "G3 (no restoring, Holling III)", "R2": "R2 (restoring, Holling II)",
          "R3": "R3 (restoring, Holling III)"}

REGIONS = ["NHL", "NA", "AB"]
REGION_TITLES = {"NHL": "Northern high latitudes (>50N)", "NA": "North Atlantic (0-50N)",
                 "AB": "Atlantic basin"}
# largest to smallest, as in the polar reference figure (BAC last)
PFTS = ["GEL", "CRU", "PTE", "MES", "PRO", "PHA", "DIA", "MIX", "COC", "FIX", "PIC", "BAC"]
ZOO = {"GEL", "CRU", "PTE", "MES", "PRO"}
LABEL_STYLE = {"zoo": ("black", "white"), "phy": ("#D3D3D3", "black"), "bac": ("#696969", "white")}
MOLC_G = 12.011e-15  # mol C -> Pg C


def pft_kind(p):
    return "bac" if p == "BAC" else ("zoo" if p in ZOO else "phy")


# ---------------------------------------------------------------- read
def read_model(model):
    """Return {region: array(len(PFTS))} in PgC, window mean of annual means; or None."""
    files = []
    for y in range(Y0, Y1 + 1):
        f = os.path.join(CLIMS, model, "timeseries",
                         "%s_%s_%04d_regmean_weekly.nc" % (model, FT, y))
        if os.path.exists(f):
            files.append((y, f))
    got = [y for y, _ in files]
    if not files:
        print("  %-16s no %s files in %d-%d" % (model, FT, Y0, Y1))
        return None
    per_year = {r: [] for r in REGIONS}
    for y, f in files:
        with nc.Dataset(f) as d:
            reg = [str(r).strip() for r in d.variables["region"][:]]
            area = np.asarray(d.variables["region_area_m2"][:], dtype=float)
            for p in PFTS:
                if p + "_int" not in d.variables:
                    ints = sorted(v for v in d.variables if v.endswith("_int"))
                    raise KeyError("%s_int not in %s; *_int variables found: %s" % (p, f, ints))
            for r in REGIONS:
                if r not in reg:
                    raise KeyError("region %s not in %s (found %s)" % (r, f, reg))
                i = reg.index(r)
                vals = []
                for p in PFTS:
                    a = np.ma.filled(d.variables[p + "_int"][:].astype(float), np.nan)
                    vals.append(np.nanmean(a[:, i]) * area[i] * MOLC_G)
                per_year[r].append(vals)
    note = "" if len(got) == Y1 - Y0 + 1 else "   <-- only %d of %d years" % (len(got), Y1 - Y0 + 1)
    print("  %-16s years found: %s%s" % (model, got, note))
    return {r: np.nanmean(np.array(per_year[r]), axis=0) for r in REGIONS}


print("CLIMS = %s   window = %d-%d" % (CLIMS, Y0, Y1))
data = {}
for m in MODELS:
    r = read_model(m)
    if r is not None:
        data[m] = r
missing = [m for m in MODELS if m not in data]
if missing:
    print("MISSING models (not plotted): " + ", ".join(missing))
if not data:
    raise SystemExit("no data found, check CLIMS / WINDOW")

# ---------------------------------------------------------------- plot
def make_figure(keys, tag):
    nk = len(keys)
    w = 0.8 / nk
    fig, axs = plt.subplots(len(REGIONS), 1, figsize=(7.2, 7.0), sharex=True)
    x = np.arange(len(PFTS))
    for ax, reg in zip(axs, REGIONS):
        for k_i, key in enumerate(keys):
            xo = x + (k_i - (nk - 1) / 2.0) * w
            for ah in ("A", "H"):
                m = [mm for mm, (ck, a) in MODELS.items() if ck == key and a == ah][0]
                if m not in data:
                    continue
                v = data[m][reg]
                if ah == "A":
                    ax.bar(xo, v, width=w * 0.92, color=colors[key], lw=0, zorder=2)
                else:
                    ax.plot(xo, v, ls="none", marker="o", ms=3.2, mfc="white",
                            mec=colors[key], mew=1.0, zorder=3)
        ax.set_title(REGION_TITLES[reg], loc="left")
        ax.set_ylabel("PgC")
        if LOGY:
            ax.set_yscale("log")
        ax.set_xlim(-0.6, len(PFTS) - 0.4)
        for b in x[:-1] + 0.5:                      # separators between PFT groups
            ax.axvline(b, color="0.8", lw=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(PFTS)
        ax.tick_params(axis="x", labelbottom=True)  # PFT names on every panel
        for lab, p in zip(ax.get_xticklabels(), PFTS):
            bg, fg = LABEL_STYLE[pft_kind(p)]
            lab.set_color(fg)
            lab.set_bbox(dict(facecolor=bg, edgecolor="none", pad=1.5))
    handles = [Line2D([], [], color=colors[k], lw=5, label=LABELS[k]) for k in keys]
    handles += [Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="0.3", mew=1.0,
                       label="Hosing 0.3 Sv (H), open marker"),
                Line2D([], [], ls="none", marker="s", ms=5, color="0.3",
                       label="No hosing (A), filled bar")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Plankton biomass, full column, window %d-%d" % (Y0, Y1), fontsize=8, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    for ext in ("png", "pdf"):
        out = os.path.join(PLTS, "B3_bars_%s_%d-%d.%s" % (tag, Y0, Y1, ext))
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


make_figure(["old", "G2", "G3", "R2", "R3"], "all")
make_figure(["old", "G2", "G3"], "oldG")
