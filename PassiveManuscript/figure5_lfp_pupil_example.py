#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:15:53 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib.patches import Rectangle
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
from serotonin_functions import (paths, load_passive_opto_times, query_ephys_sessions,
                                 load_lfp, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
SUBJECT = 'ZFM-02600'
DATE = '2021-08-25'
PROBE = 'probe00'
PRE_TIME = 1
POST_TIME = 2
LFP_CHAN = 100

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Query sessions
rec = query_ephys_sessions(one=one)
eid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'eid'].values[0]
pid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'pid'].values[0]

# Load in LFP and laser pulses
opto_train_times, _ = load_passive_opto_times(eid, one=one)
lfp, time = load_lfp(eid, PROBE, time_start=opto_train_times[0]-10,
                     time_end=opto_train_times[-1]+10,
                     relative_to='begin', one=one)

# Get pupil diameter
video_times, XYs = get_dlc_XYs(one, eid)
print('Calculating smoothed pupil trace')
raw_diameter, diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)
diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                 / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.75), dpi=dpi)

ax1.add_patch(Rectangle((-0.5, -5), 0.5, 10, color=[.5, .5, .5], alpha=0.25, lw=0))
ax1.add_patch(Rectangle((0, -5), 1, 10, color='royalblue', alpha=0.25, lw=0))

plot_lfp = zscore(lfp[LFP_CHAN, (time > opto_train_times[0] - PRE_TIME)
                      & (time < opto_train_times[0] + POST_TIME)]) + 2
plot_time = np.linspace(-PRE_TIME, POST_TIME, plot_lfp.shape[0])
ax1.plot(plot_time, plot_lfp, color=[.3, .3, .3])

plot_pupil = zscore(diameter[(video_times > opto_train_times[0] - PRE_TIME)
                             & (video_times < opto_train_times[0] + POST_TIME)]) - 2
plot_time = np.linspace(-PRE_TIME, POST_TIME, plot_pupil.shape[0])
ax1.plot(plot_time, plot_pupil, color=colors['general'])

ax1.axis('off')
plt.tight_layout()
plt.savefig(join(fig_path, 'example_pupil_lfp.pdf'))


