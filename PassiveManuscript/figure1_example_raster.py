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
one = ONE()

eid = '2f72e869-fbd1-4ecd-a503-30dc24980ecf'
probe = 'probe00'
subject = 'ZFM-01802'
date = '2021-03-09'

spikes = one.load_object(eid, obj='spikes', collection=f'alf/{probe}')

opto_times, _ = load_passive_opto_times(eid, one=one)

# compute raster map as a function of site depth
iok = ~np.isnan(spikes.depths)
R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], 0.01, 20, weights=None)
f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)
ax.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
          extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
ax.set(title=f'{subject} {date} {probe}', xlim=[2430.32674302 - 6, 2430.32674302 + 5.5],
       xticks=[2430.32674302, 2430.32674302+ 1])
plt.tight_layout()
sns.despine(trim=True, offset=4)
plt.savefig('/home/guido/Figures/PaperPassive/figure1_example_raster.pdf')