#!/usr/bin/env python
"""MLD maps on the raw ORCA2 grid (no projection, no regridding, no climatology files).
Reads mldr10_1 straight from the run's weekly grid_T files, makes 5-yr means of
  a) the annual mean (all 52 weeks)   b) the March mean (weeks whose centre date is in March)
Rows = HA01 HH01 G2A2 G2H2 R2A2 R2H2; columns = 1940-44 | 2015-19 | late - early.
Writes, for the whole grid and for a North Atlantic box:
  plts/mld_{annual,march}_maps.{png,pdf}   and   plts/mld_{annual,march}_maps_NAtl.{png,pdf}
Box in grid indices (python slices, ends clipped to the grid; tick labels on the maps are grid indices):
  python 05_mld_maps.py --x 100 160 --y 80 148
"""
import os, argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import seaborn as sns
try:
    import cmocean; cmap_mld = cmocean.cm.deep
except ImportError:
    cmap_mld = 'GnBu'

ap = argparse.ArgumentParser()
ap.add_argument('--x', nargs=2, type=int, default=[100, 160], help='North Atlantic box, x index start end')
ap.add_argument('--y', nargs=2, type=int, default=[80, 148],  help='North Atlantic box, y index start end')
args = ap.parse_args()

RUN_ROOTS = [Path(p) for p in (os.environ.get('RUNS', '/gpfs/data/greenocean/software/runs/NEMO5'),
                               '/gpfs/afm/greenocean/software/runs/NEMO5')]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plts'; OUT.mkdir(exist_ok=True)

rows = ['TOM12_TJ_HA01', 'TOM12_TJ_HH01', 'TOM12_TJ_G2A2', 'TOM12_TJ_G2H2', 'TOM12_TJ_R2A2', 'TOM12_TJ_R2H2']
early, late = (1940, 1944), (2015, 2019)
VAR = 'mldr10_1'
MLD_MAX  = {'annual': None, 'march': None}     # m; None = auto (99.5th percentile in the plotted box, rounded up)
DIFF_MAX = {'annual': None, 'march': None}
cmap_diff = 'PuOr_r'                           # positive (deeper later) = orange
LAND = '#B49A7C'                               # land colour (tan brown)
FIG_MAX = (7.5, 11.5)                          # max figure size in inches (width, height); panels are packed to fill it
colors = {'G2': '#3A5B8C', 'R2': '#B23D32', 'old': 'black'}

font = 'DejaVu Sans'
for f in (ROOT / 'fonts').glob('Jost-*.ttf'):
    if 'Variable' not in f.name: fm.fontManager.addfont(str(f)); font = 'Jost'
sns.set_theme(style='ticks', rc={'font.family': font})

def year_means(run, year):
    """(annual mean, March mean) of mldr10_1 for one year, shape (y, x); None if file missing."""
    name = f'ORCA2_7d_{year}0101_{year}1231_grid_T.nc'
    f = next((r / run / name for r in RUN_ROOTS if (r / run / name).exists()), None)
    if f is None:
        print(f'[{run}] {year}: not found ({name} in {[str(r / run) for r in RUN_ROOTS]})'); return None
    with xr.open_dataset(f, decode_times=False) as ds:
        v = ds[VAR].values.astype(float)                                    # (time, y, x)
        t = (ds['time_centered'] if 'time_centered' in ds else ds['time_counter']).values
    sec = np.datetime64('1900-01-01', 's') + t.astype('int64').astype('timedelta64[s]')   # seconds since 1900-01-01
    month = sec.astype('datetime64[M]').astype(int) % 12 + 1
    return v.mean(0), (v[month == 3].mean(0) if (month == 3).any() else None)

def window(run, win):
    ys = [year_means(run, y) for y in range(win[0], win[1] + 1)]
    out = {}
    for k, name in enumerate(('annual', 'march')):
        arrs = [y[k] for y in ys if y is not None and y[k] is not None]
        if arrs:
            m = np.mean(arrs, axis=0); m[m == 0] = np.nan                    # land = 0 -> land colour
            out[name] = (m, len(arrs))
    print(f'[{run}] {win[0]}-{win[1]}: {sum(y is not None for y in ys)}/{win[1] - win[0] + 1} files')
    return out

def nice(v):
    e = 10 ** np.floor(np.log10(v))
    return next(m * e for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v)

def ticks(a, b):
    step = 20 if b - a > 100 else 10 if b - a > 40 else 5
    return np.arange(int(np.ceil(a / step)) * step, b, step)

data = {r: (window(r, early), window(r, late)) for r in rows}
ny, nx = next(v[0].shape for r in rows for w in data[r] for v in w.values())

def draw(tag, desc, xb, yb, suffix):
    x0, x1 = max(0, xb[0]), min(nx, xb[1]); y0, y1 = max(0, yb[0]), min(ny, yb[1])
    sl = (slice(y0, y1), slice(x0, x1))
    crop = lambda r: None if r is None else (r[0][sl], r[1])
    E = {r: crop(data[r][0].get(tag)) for r in rows}; L = {r: crop(data[r][1].get(tag)) for r in rows}
    D = {r: (L[r][0] - E[r][0], min(L[r][1], E[r][1])) if E[r] and L[r] else None for r in rows}
    allv = [a[0][np.isfinite(a[0])] for a in list(E.values()) + list(L.values()) if a]
    alld = [np.abs(d[0][np.isfinite(d[0])]) for d in D.values() if d]
    if not allv: print(f'{tag}{suffix}: no data at all, skipping'); return
    vmax = MLD_MAX[tag] or nice(np.percentile(np.concatenate(allv), 99.5))
    dmax = DIFF_MAX[tag] or (nice(np.percentile(np.concatenate(alld), 99.5)) if alld else 1)
    print(f'{tag}{suffix}: x {x0}-{x1}, y {y0}-{y1}; MLD colour range 0-{vmax:g} m, difference +-{dmax:g} m')

    # exact packing: panels fill the allowed figure size (inches)
    LM, RM, TM, BM, gx, gy = 0.80, 0.40, 0.50, 1.20, 0.07, 0.07
    a = (x1 - x0) / (y1 - y0)
    h = min((FIG_MAX[0] - LM - RM - 2 * gx) / 3 / a, (FIG_MAX[1] - TM - BM - 5 * gy) / 6); w = h * a
    W, H = LM + RM + 3 * w + 2 * gx, TM + BM + 6 * h + 5 * gy
    fig = plt.figure(figsize=(W, H))
    axs = np.empty((6, 3), dtype=object)
    for i in range(6):
        for j in range(3):
            axs[i, j] = fig.add_axes([(LM + j * (w + gx)) / W, 1 - (TM + (i + 1) * h + i * gy) / H, w / W, h / H])
    mp = {}
    for i, run in enumerate(rows):
        key = 'old' if 'HA01' in run or 'HH01' in run else run[9:11]
        for j, res in enumerate((E[run], L[run], D[run])):
            ax = axs[i, j]; ax.set_facecolor(LAND)
            ax.set_xticks(ticks(x0, x1) if i == 5 else []); ax.set_yticks(ticks(y0, y1) if j == 2 else [])
            ax.tick_params(labelsize=6, length=2, pad=1); ax.yaxis.tick_right()
            ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y0 - 0.5, y1 - 0.5)
            if res is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='white'); continue
            kw = dict(vmin=-dmax, vmax=dmax, cmap=cmap_diff) if j == 2 else dict(vmin=0, vmax=vmax, cmap=cmap_mld)
            mp[j] = ax.imshow(np.ma.masked_invalid(res[0]), origin='lower', aspect='auto', interpolation='nearest',
                              extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5), **kw)
            if res[1] < 5: ax.text(0.98, 0.04, f'{res[1]} yr', ha='right', transform=ax.transAxes, fontsize=6, color='k')
        axs[i, 0].text(-0.03, 0.5, run.replace('TOM12_TJ_', ''), transform=axs[i, 0].transAxes, rotation=90,
                       ha='right', va='center', color=colors[key], fontweight='semibold', fontsize=8)
    for j, t in enumerate([f'{early[0]}–{early[1]}', f'{late[0]}–{late[1]}', f'{late[0]}–{late[1]}\nminus {early[0]}–{early[1]}']):
        axs[0, j].set_title(t, fontsize=9, fontweight='semibold', pad=3)
    cy, ch = 0.80 / H, 0.13 / H
    if 0 in mp:
        cax = fig.add_axes([(LM + 0.1 * (2 * w + gx)) / W, cy, 0.8 * (2 * w + gx) / W, ch])
        fig.colorbar(mp[0], cax=cax, orientation='horizontal', extend='max').set_label(f'MLD, {desc} (m)')
    if 2 in mp:
        cax = fig.add_axes([(LM + 2 * (w + gx) + 0.1 * w) / W, cy, 0.8 * w / W, ch])
        fig.colorbar(mp[2], cax=cax, orientation='horizontal', extend='both').set_label('difference (m)')
    for r0, txt, col in [(0, 'OLD RUNSET', colors['old']), (2, 'NO SALINITY RESTORING', colors['G2']),
                         (4, 'YES SALINITY RESTORING', colors['R2'])]:
        top, bot = axs[r0, 0].get_position().y1, axs[r0 + 1, 0].get_position().y0
        fig.text(0.13 / W, (top + bot) / 2, txt, rotation=90, ha='center', va='center', color=col, fontweight='semibold', fontsize=7.5)
        fig.add_artist(Line2D([0.30 / W] * 2, [bot + 0.004, top - 0.004], color=col, lw=1.2))
    fig.text(LM / W, 0.06 / H, f'Raw ORCA2 grid, axes = grid indices (x right, y up); {VAR}, 5-yr mean of the {desc}.', fontsize=7, color='grey')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'mld_{tag}_maps{suffix}.{ext}', dpi=300)
    plt.close(fig)
    print('wrote', OUT / f'mld_{tag}_maps{suffix}.png')

for tag, desc in (('annual', 'yearly mean'), ('march', 'March mean')):
    draw(tag, desc, (0, nx), (0, ny), '')
    draw(tag, desc, args.x, args.y, '_NAtl')
