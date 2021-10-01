#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, load_opto_times
from one.api import ONE
one = ONE()

# Settings
SUBJECT = 'ZFM-02600'
DATE = '2021-08-26'
PROBE = 'probe01'
NEURON = 559
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.05
_, fig_path, save_path = paths()

# Query session
eid = one.search(subject=SUBJECT, date_range=DATE)[0]

# Load in laser pulse times
opto_train_times = load_opto_times(eid, one=one)

# Load in spikes
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

# Plot PSTH
colors, dpi = figure_style()
p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
peri_event_time_histogram(spikes[PROBE].times, spikes[PROBE].clusters, opto_train_times,
                          NEURON, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                          include_raster=True, error_bars='sem', ax=ax,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3},
                          raster_kwargs={'color': 'black', 'lw': 0.3},
                          eventline_kwargs={'lw': 0})
ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
ax.plot([0, 1], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                 ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=2, color='royalblue')
ax.set(ylabel='spikes/s', xlabel='Time (s)',
       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()

