#!/usr/bin/env python
"""
B1_timeseries.py  --  annual-mean biology/carbon timeseries, regions x variables.

Layout: 4 rows (PPINT, EXP100, TChl, Cflx) x 3 columns (NHL, NA, AB).
Input : <CLIMS>/<model>/timeseries/<model>_diad_T_<YYYY>_regmean_weekly.nc
        (weekly, area-weighted regional means on the native grid; dims time_counter, region)
Output: <PLTS>/B1_timeseries.{png,pdf} (all runs) and B1_timeseries_G.{png,pdf} (no restoring)

Env overrides:
  CLIMS  clims root   (default /gpfs/scratch/greenocean/mep22dku/clims)
  PLTS   plot folder  (default ../plts relative to this script)
  FONTS  font folder  (default ../fonts relative to this script)
  UNITS  'area' (default): per-m2 means (mol C m-2 yr-1, mgChl m-3)
         'total'        : region sums, PgC yr-1 for PPINT/EXP100/Cflx (mean x region_area_m2)
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
UNITS = os.environ.get("UNITS", "area")
FT = "diad_T"
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
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

colors = {"G2": "#0072B2", "G3": "#56B4E9", "R2": "#B23D32", "R3": "#F06C1B", "old": "black"}
linestyle = {"A": "-", "H": ":"}

# model name -> (colour key, A/H)
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
SEC_YR = 365.0 * 86400.0   # noleap calendar (per the file header)
MOLC_G = 12.011e-15        # mol C -> Pg C

# variable -> (label, factor to convert file units, is_flux_like)
# file units: PPINT, EXP100, Cflx mol/m2/s ; TChl mgChl/m3 (surface)
VARS = [
    ("PPINT",  "Primary production", SEC_YR, True),
    ("EXP100", "Export at 100 m",    SEC_YR, True),
    ("TChl",   "Surface chlorophyll", 1.0,   False),
    ("Cflx",   "Air-sea CO$_2$ flux (into ocean +)", SEC_YR, True),
]


def ylabel(var, flux):
    if not flux:
        return "mg Chl m$^{-3}$"
    return "PgC yr$^{-1}$" if UNITS == "total" else "mol C m$^{-2}$ yr$^{-1}$"


# ---------------------------------------------------------------- read
def read_model(model):
    """Return dict var -> {region: (years, annual_mean)} or None."""
    files = sorted(glob.glob(os.path.join(CLIMS, model, "timeseries",
                                          "%s_%s_*_regmean_weekly.nc" % (model, FT))))
    if not files:
        print("  %-16s no files found" % model)
        return None
    years, store = [], {v[0]: {r: [] for r in REGIONS} for v in VARS}
    nweeks = []
    for f in files:
        m = re.search(r"_(\d{4})_regmean_weekly\.nc$", f)
        if not m:
            continue
        y = int(m.group(1))
        with nc.Dataset(f) as d:
            reg = [str(r).strip() for r in d.variables["region"][:]]
            area = np.asarray(d.variables["region_area_m2"][:], dtype=float)
            for r in REGIONS:
                if r not in reg:
                    raise KeyError("region %s not in %s (found %s)" % (r, f, reg))
            for var, _, fac, flux in VARS:
                a = np.ma.filled(d.variables[var][:].astype(float), np.nan)  # (time, region)
                nweeks.append(a.shape[0])
                for r in REGIONS:
                    i = reg.index(r)
                    x = np.nanmean(a[:, i]) * fac      # equal-weight mean of 7-day records
                    if UNITS == "total" and flux:
                        x = x * area[i] * MOLC_G
                    store[var][r].append(x)
        years.append(y)
    years = np.array(years)
    out = {v: {r: (years, np.array(store[v][r])) for r in REGIONS} for v in store}
    print("  %-16s %3d files, %d-%d, weeks/file min %d max %d"
          % (model, len(years), years.min(), years.max(), min(nweeks), max(nweeks)))
    return out


print("CLIMS = %s   UNITS = %s" % (CLIMS, UNITS))
data = {}
for model in MODELS:
    r = read_model(model)
    if r is not None:
        data[model] = r
missing = [m for m in MODELS if m not in data]
if missing:
    print("MISSING models (not plotted): " + ", ".join(missing))
if not data:
    raise SystemExit("no data found, check CLIMS")

# ---------------------------------------------------------------- plot
INCLUDE_OLD_IN_G = True   # old runset (HA01/HH01) also has no restoring; set False for G2/G3 only


def make_figure(models, keys, tag):
    models = [m for m in models if m in data]
    fig, axs = plt.subplots(len(VARS), len(REGIONS), figsize=(7.2, 7.4), sharex=True)
    for j, reg in enumerate(REGIONS):
        for i, (var, name, fac, flux) in enumerate(VARS):
            ax = axs[i, j]
            for model in models:
                ck, ah = MODELS[model]
                x, y = data[model][var][reg]
                ax.plot(x + 0.5, y, color=colors[ck], ls=linestyle[ah], lw=1.1)
            ax.axvline(1940, color="0.6", lw=0.6, ls="--", zorder=0)
            if var == "Cflx":
                ax.axhline(0, color="0.6", lw=0.5, zorder=0)
            if i == 0:
                ax.set_title(REGION_TITLES[reg])
            if j == 0:
                ax.set_ylabel("%s\n%s" % (name, ylabel(var, flux)))
            if i == len(VARS) - 1:
                ax.set_xlabel("Year")
            ax.set_xlim(1920, 2025)
    handles = [Line2D([], [], color=colors[k], lw=1.5, label=LABELS[k]) for k in keys]
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.5, label="No hosing (A)"),
                Line2D([], [], color="0.3", ls=":", lw=1.5, label="Hosing 0.3 Sv from 1940 (H)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(PLTS, "B1_timeseries%s.%s" % (tag, ext))
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, "(%d models)" % len(models))
    plt.close(fig)


# 1) all runs
make_figure(list(MODELS), list(LABELS), "")

# 2) no salinity restoring only (G runs, plus old runset as reference)
g_keys = ["G2", "G3"] + (["old"] if INCLUDE_OLD_IN_G else [])
g_models = [m for m, (ck, ah) in MODELS.items() if ck in g_keys]
make_figure(g_models, g_keys, "_G")
