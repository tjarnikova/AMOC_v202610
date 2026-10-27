#!/usr/bin/env python
"""
B2_polar_biomass.py  --  polar ("radar") plot of regional depth-integrated PFT biomass.

One figure = 1 row x 3 polar panels (NHL, NA, AB); one line per run.
Axes (clockwise from top, as in the reference figure):
    BAC GEL CRU PTE MES PRO PHA DIA MIX COC FIX PIC
Value = regional total in PgC = mean(<name>_int, mol C/m2) x region_area_m2 x 12.011e-15,
averaged over a window of annual means. Biomass is FULL COLUMN (not 0-200 m).

Input : <CLIMS>/<model>/timeseries/<model>_ptrc_T_<YYYY>_regmean_weekly.nc
        (variables <name>_int, dims time_counter x region; region_area_m2)
Output: <PLTS>/B2_polar_<set>_<y0>-<y1>.{png,pdf}   set = all | oldG

Env overrides:
  CLIMS   clims root   (default /gpfs/scratch/greenocean/mep22dku/clims)
  PLTS    plot folder  (default ../plts relative to this script)
  FONTS   font folder  (default ../fonts relative to this script)
  WINDOW  'y0-y1'      (default 2015-2019, the 'late' window; early = 1940-1944)
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
# clockwise from top, as in the reference figure
PFTS = ["BAC", "GEL", "CRU", "PTE", "MES", "PRO", "PHA", "DIA", "MIX", "COC", "FIX", "PIC"]
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
theta = np.linspace(0, 2 * np.pi, len(PFTS), endpoint=False)


def make_figure(models, keys, tag):
    models = [m for m in models if m in data]
    fig, axs = plt.subplots(1, len(REGIONS), figsize=(7.2, 3.3),
                            subplot_kw=dict(projection="polar"))
    for ax, reg in zip(axs, REGIONS):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for m in models:
            ck, ah = MODELS[m]
            v = data[m][reg]
            ax.plot(np.r_[theta, theta[0]], np.r_[v, v[0]], color=colors[ck],
                    ls=linestyle[ah], lw=1.1)
        ax.set_xticks(theta)
        ax.set_xticklabels([])
        ax.set_rmin(0)
        ax.set_rlabel_position(15)
        ax.grid(color="0.8", lw=0.5)
        rmax = ax.get_ylim()[1]
        for t, p in zip(theta, PFTS):
            bg, fg = LABEL_STYLE[pft_kind(p)]
            ax.text(t, rmax * 1.13, p, ha="center", va="center", fontsize=6, color=fg,
                    bbox=dict(facecolor=bg, edgecolor="none", pad=1.5))
        ax.set_title(REGION_TITLES[reg], pad=16)
    axs[0].text(-0.2, 0.5, "Plankton biomass, full column (PgC)", rotation=90,
                transform=axs[0].transAxes, ha="center", va="center")
    handles = [Line2D([], [], color=colors[k], lw=1.5, label=LABELS[k]) for k in keys]
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.5, label="No hosing (A)"),
                Line2D([], [], color="0.3", ls=":", lw=1.5, label="Hosing 0.3 Sv (H)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.17, wspace=0.35)
    for ext in ("png", "pdf"):
        out = os.path.join(PLTS, "B2_polar_%s_%d-%d.%s" % (tag, Y0, Y1, ext))
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, "(%d models)" % len(models))
    plt.close(fig)


make_figure(list(MODELS), list(LABELS), "all")
g_keys = ["G2", "G3", "old"]
make_figure([m for m, (ck, ah) in MODELS.items() if ck in g_keys], g_keys, "oldG")
