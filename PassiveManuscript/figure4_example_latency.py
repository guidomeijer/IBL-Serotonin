#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from matplotlib.patches import Rectangle
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, load_passive_opto_times
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings


SUBJECT = 'ZFM-03330'
DATE = '2022-02-16'
PROBE = 'probe00'
NEURON = 209
TITLE = 'Ex. thalamus neuron'
"""
SUBJECT = 'ZFM-03330'
DATE = '2022-02-15'
PROBE = 'probe00'
NEURON = 323
TITLE = 'Ex. frontal cortex neuron'

SUBJECT = 'ZFM-04122'
DATE = '2022-05-12'
PROBE = 'probe00'
NEURON = 265
TITLE = 'Ex. superior colliculus neuron'
"""

T_BEFORE = 1  # for plotting
T_AFTER = 2
ZETA_BEFORE = 0  # baseline period to include for zeta test
PRE_TIME = [1, 0]  # for modulation index
POST_TIME = [0, 1]
BIN_SIZE = 0.05
SMOOTHING = 0.025
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Get session details
ins = one.alyx.rest('insertions', 'list', date=DATE, subject=SUBJECT, name=PROBE)
pid = ins[0]['id']
eid = ins[0]['session']

# Get peak latency from file
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
this_neuron = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['neuron_id'] == NEURON)]
latency = all_neurons.loc[(all_neurons['pid'] == pid) & (all_neurons['neuron_id'] == NEURON),
                          'latency_peak_onset'].values[0]

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# %% Plot PSTH
colors, dpi = figure_style()
p, ax = plt.subplots(1, 1, figsize=(1.75, 2), dpi=dpi)
ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))

peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                          NEURON, t_before=T_BEFORE, t_after=T_AFTER, smoothing=SMOOTHING,
                          bin_size=BIN_SIZE, include_raster=True, error_bars='sem', ax=ax,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                          raster_kwargs={'color': 'black', 'lw': 0.3},
                          eventline_kwargs={'lw': 0})
ax.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)', title=TITLE,
       yticks=[np.round(ax.get_ylim()[1])],
       ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
# ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

peths, _ = calculate_peths(spikes.times, spikes.clusters, [NEURON],
                           opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
peak_ind = np.argmin(np.abs(peths['tscale'] - latency))
peak_act = peths['means'][0][peak_ind]
ax.plot([latency, latency], [peak_act, peak_act], 'x', color='red', lw=2)
#ax.plot([latency, latency], [peak_act, 14], ls='--', color='red', lw=0.5)
ax.text(latency-0.7, 6.2, f'{latency*1000:.0f} ms', color='red', va='center', ha='left', fontsize=5)

plt.tight_layout()

plt.savefig(join(fig_path, f'latency_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))

