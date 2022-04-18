#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:46:00 2022
By: Guido Meijer
"""

import numpy as np
from one.api import ONE
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import load_passive_opto_times
from brainbox.ephys_plots import scatter_raster_plot
from brainbox.plot_base import plot_scatter
from brainbox.processing import bincount2D
from brainbox.plot import driftmap
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

#eid = '0d24afce-9d3c-449e-ac9f-577eefefbd7e'
eid = '0d24afce-9d3c-449e-ac9f-577eefefbd7e'

opto_times, _ = load_passive_opto_times(eid, one=one)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=600)

sl = SpikeSortingLoader(eid=eid, pname='probe00', one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
iok = ~np.isnan(spikes.depths)
R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], 0.01, 20, weights=None)

ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
          extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
ax1.set(xlim=[opto_times[0] - 6, opto_times[0] + 5.5], ylim=[0, 4000],
       xticks=[opto_times[0], opto_times[0]+ 1], xticklabels=['',''])

sl = SpikeSortingLoader(eid=eid, pname='probe01', one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
iok = ~np.isnan(spikes.depths)
R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], 0.01, 20, weights=None)

ax2.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
          extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
ax2.set(xlim=[opto_times[0] - 6, opto_times[0] + 5.5], ylim=[0,4000],
       xticks=[opto_times[0], opto_times[0]+ 1], xticklabels=['',''])

f.suptitle(f'{eid}')

plt.tight_layout()
sns.despine(trim=True, offset=4)
plt.savefig('/home/guido/Figures/PaperPassive/figure1_example_raster.pdf')
plt.savefig('/home/guido/Dropbox/Work/PaperPassive/figure1_example_raster.pdf')
