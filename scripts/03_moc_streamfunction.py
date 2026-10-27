#!/usr/bin/env python
"""Atlantic overturning streamfunction (zomsfatl): rows = HA01 HH01 G2A2 G2H2 R2A2 R2H2,
columns = early 5-yr mean | late 5-yr mean | late - early.   Writes plts/moc_streamfunction.{png,pdf}
Files are searched in MOC_DIR (flat, as in MF_AMOC_MT.py), then in RUNS/<run>/. Missing years are skipped.
"""
import os, glob
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import seaborn as sns
try:
    import cmocean; cmap_psi = cmocean.cm.balance
except ImportError:
    cmap_psi = 'RdBu_r'

MOC_DIR = Path(os.environ.get('MOC_DIR', '/gpfs/data/greenocean/software/resources/CDFTOOLS/MOCresults'))
RUNS    = Path(os.environ.get('RUNS',    '/gpfs/data/greenocean/software/runs/NEMO5'))
MASK    = 'MTbasins'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plts'; OUT.mkdir(exist_ok=True)

rows = ['TOM12_TJ_HA01', 'TOM12_TJ_HH01', 'TOM12_TJ_G2A2', 'TOM12_TJ_G2H2', 'TOM12_TJ_R2A2', 'TOM12_TJ_R2H2']
early, late = (1940, 1944), (2015, 2019)
PSI_MAX, DIFF_MAX = 16, 9                 # Sv (colour range)
lev_psi  = np.linspace(-PSI_MAX, PSI_MAX, 2 * PSI_MAX + 1)
lev_line = np.arange(-PSI_MAX, PSI_MAX + 1, 4)
lev_diff = np.linspace(-DIFF_MAX, DIFF_MAX, 2 * DIFF_MAX + 1)
cmap_diff = 'PuOr_r'                       # different diverging map, red-green safe
DEPTH_MAX, LAT_MIN, LAT_MAX = 3000, -30, 75
colors = {'G2': '#3A5B8C', 'R2': '#B23D32', 'old': 'black'}

font = 'DejaVu Sans'
for f in (ROOT / 'fonts').glob('Jost-*.ttf'):
    if 'Variable' not in f.name: fm.fontManager.addfont(str(f)); font = 'Jost'
sns.set_theme(style='ticks', rc={'font.family': font})

def find(run, y):
    pat = f'*{run}_7d_{y}0101_{y}1231_MOC_{MASK}.nc'
    for d in (MOC_DIR, RUNS):
        m = glob.glob(str(d / pat)) or glob.glob(str(d / run / pat))
        if m: return m[0]

def psi(run, y0, y1):
    """5-yr mean psi (depth, lat), lat, depth (positive down), n files; None if no files."""
    fs = [f for f in (find(run, y) for y in range(y0, y1 + 1)) if f]
    print(f'[{run}] {y0}-{y1}: {len(fs)}/{y1 - y0 + 1} files')
    if not fs:
        print(f'   searched {MOC_DIR} and {RUNS / run}'); return None
    ps = []
    for f in fs:
        with xr.open_dataset(f) as ds:
            da = ds['zomsfatl'].squeeze('x', drop=True)
            ps.append(da.mean('time_counter').load().values)
            lat   = ds['nav_lat'].squeeze('x', drop=True).values
            depth = np.abs(ds['depthw'].values)           # depthw may be stored negative
    p = np.mean(ps, axis=0)
    if np.nanmax(np.abs(p)) > 1e3: p = p * 1e-6           # m3/s -> Sv
    return p, lat, depth, len(fs)

fig, axs = plt.subplots(len(rows), 3, figsize=(7.2, 9.2), sharex=True, sharey=True, constrained_layout=True)
mp = {}
for i, run in enumerate(rows):
    e, l = psi(run, *early), psi(run, *late)
    key = 'old' if 'HA01' in run or 'HH01' in run else run[9:11]
    for j in range(3):
        ax = axs[i, j]
        if j == 2: res = (l[0] - e[0], l[1], l[2], min(e[3], l[3])) if e and l else None
        else:      res = (e, l)[j]
        if res is None:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='grey'); continue
        lev, cm = (lev_diff, cmap_diff) if j == 2 else (lev_psi, cmap_psi)
        mp[j] = ax.contourf(res[1], res[2], res[0], levels=lev, cmap=cm, extend='both')
        if j < 2:
            ax.contour(res[1], res[2], res[0], levels=lev_line, colors='k', linewidths=0.5,
                       linestyles=['dashed' if v < 0 else 'solid' for v in lev_line])
        if res[3] < 5: ax.text(0.98, 0.04, f'{res[3]} yr', ha='right', transform=ax.transAxes, fontsize=6, color='grey')
    axs[i, 0].set_ylabel(f'{run.replace("TOM12_TJ_", "")}\ndepth (m)', color=colors[key], fontweight='semibold', fontsize=8)
for j, t in enumerate([f'{early[0]}–{early[1]}', f'{late[0]}–{late[1]}', f'{late[0]}–{late[1]} minus {early[0]}–{early[1]}']):
    axs[0, j].set_title(t, fontsize=9, fontweight='semibold')
for ax in axs[-1]: ax.set_xlabel('latitude (°N)')
axs[0, 0].set_ylim(DEPTH_MAX, 0); axs[0, 0].set_xlim(LAT_MIN, LAT_MAX)
if 0 in mp:
    cb = fig.colorbar(mp[0], ax=axs[:, :2], location='bottom', shrink=0.55, aspect=40, label='Atlantic streamfunction (Sv)')
    cb.set_ticks(np.arange(-PSI_MAX, PSI_MAX + 1, 4))
if 2 in mp:
    cb2 = fig.colorbar(mp[2], ax=axs[:, 2], location='bottom', shrink=0.9, aspect=20, label='difference (Sv)')
    cb2.set_ticks(np.arange(-DIFF_MAX, DIFF_MAX + 1, 3))
# group labels (left of the run labels): old runset | no restoring (G) | restoring (R)
fig.canvas.draw()                                            # finalise layout (no get_layout_engine: old matplotlib)
xmin = min(fig.transFigure.inverted().transform(ax.get_tightbbox(fig.canvas.get_renderer()).get_points())[0, 0]
           for ax in axs[:, 0])
xbar, xtxt = xmin - 0.012, xmin - 0.030                      # just left of the run labels; saved with bbox_inches='tight'
for r0, txt, col in [(0, 'OLD RUNSET', colors['old']), (2, 'NO SALINITY RESTORING', colors['G2']),
                     (4, 'YES SALINITY RESTORING', colors['R2'])]:
    top, bot = axs[r0, 0].get_position().y1, axs[r0 + 1, 0].get_position().y0
    fig.text(xtxt, (top + bot) / 2, txt, rotation=90, ha='center', va='center', color=col, fontweight='semibold', fontsize=7.5)
    fig.add_artist(Line2D([xbar, xbar], [bot + 0.004, top - 0.004], color=col, lw=1.2))
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'moc_streamfunction.{ext}', dpi=300, bbox_inches='tight')
print('wrote', OUT / 'moc_streamfunction.png')
