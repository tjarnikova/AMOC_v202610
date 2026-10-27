#!/usr/bin/env python
"""AMOC at 26.5N: filtered timeseries. Top: Holling II + original runset. Bottom: + Holling III.
Writes plts/amoc_26N_timeseries.{png,pdf}.  Run from anywhere: python 02_amoc_timeseries.py
"""
import os, glob
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

CLIMS = Path(os.environ.get('CLIMS', '/gpfs/scratch/greenocean/mep22dku/clims'))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plts'; OUT.mkdir(exist_ok=True)
FILT_YR = 5                      # centred running mean (years); weekly data, 52 steps/yr

colors = {'G2': '#3A5B8C', 'G3': '#6B9BD8', 'R2': '#B23D32', 'R3': '#F06C1B', 'old': 'black'}
linestyle = {'A': '-', 'H': ':'}

font = 'DejaVu Sans'
for f in (ROOT / 'fonts').glob('Jost-*.ttf'):
    if 'Variable' not in f.name: fm.fontManager.addfont(str(f)); font = 'Jost'
sns.set_theme(style='ticks', rc={'font.family': font})

def key_ah(run):                 # TOM12_TJ_G2H2 -> ('G2','H'); HA01/HH01 -> ('old','A'/'H')
    tag = run.replace('TOM12_TJ_', '')
    return ('old', tag[1]) if tag in ('HA01', 'HH01') else (tag[:2], tag[2])

data = {}
for f in sorted(glob.glob(str(CLIMS / '*' / '*_AMOC_26N_*.nc'))):
    ds = xr.open_dataset(f)
    run = ds.attrs.get('source_model', Path(f).parent.name)
    da = ds['AMOC_26N']
    data[run] = da.rolling(time_counter=FILT_YR * 52, center=True, min_periods=FILT_YR * 26).mean()

panels = [('Holling II + original runset', {'old', 'G2', 'R2'}),
          ('Holling II + III + original runset', {'old', 'G2', 'R2', 'G3', 'R3'})]
fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.8), sharex=True, sharey=True)
for ax, (title, keys) in zip(axs, panels):
    for run, da in sorted(data.items()):
        k, ah = key_ah(run)
        if k in keys:
            ax.plot(da.time_counter, da, color=colors[k], ls=linestyle[ah], lw=1.4, label=run.replace('TOM12_TJ_', ''))
    ax.axvline(np.datetime64('1940-01-01'), color='grey', lw=0.7, ls='--')
    ax.set_title(title, loc='left', fontsize=10, fontweight='semibold')
    ax.set_ylabel('AMOC 26.5°N (Sv)')
    ax.legend(ncol=2, fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
axs[-1].set_xlabel('')
fig.text(0.01, 0.005, f'{FILT_YR}-yr running mean; dashed line = hosing start (1940).', fontsize=7, color='grey')
sns.despine(fig)
fig.tight_layout(rect=(0, 0.02, 1, 1))
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'amoc_26N_timeseries.{ext}', dpi=300)
