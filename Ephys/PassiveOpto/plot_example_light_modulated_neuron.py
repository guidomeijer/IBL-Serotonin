#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from brainbox.task.closed_loop import roc_single_event
import pandas as pd
from os import mkdir
from sklearn.metrics import roc_curve
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_ephys_sessions, load_passive_opto_times, remove_artifact_neurons
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
"""
SUBJECT = 'ZFM-01802'
DATE = '2021-03-11'
PROBE = 'probe00'
NEURON = 235
"""
SUBJECT = 'ZFM-01802'
DATE = '2021-03-09'
PROBE = 'probe00'
NEURON = 349

T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0.5, 1]
BIN_SIZE = 0.05
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons')

# Get session details
eid = one.search(subject=SUBJECT, date_range=DATE)[0]

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
    eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

# Select spikes of passive period
start_passive = opto_train_times[0] - 360
spikes[PROBE].clusters = spikes[PROBE].clusters[spikes[PROBE].times > start_passive]
spikes[PROBE].times = spikes[PROBE].times[spikes[PROBE].times > start_passive]

# Get spike counts for baseline and event timewindow
baseline_times = np.column_stack(((opto_train_times - PRE_TIME[0]), (opto_train_times - PRE_TIME[1])))
baseline_counts, cluster_ids = get_spike_counts_in_bins(spikes[PROBE].times, spikes[PROBE].clusters,
                                                        baseline_times)
times = np.column_stack(((opto_train_times + POST_TIME[0]), (opto_train_times + POST_TIME[1])))
spike_counts, cluster_ids = get_spike_counts_in_bins(spikes[PROBE].times, spikes[PROBE].clusters, times)

fpr, tpr, _ = roc_curve(np.append(np.zeros(baseline_counts.shape[1]), np.ones(baseline_counts.shape[1])),
                        np.append(baseline_counts[cluster_ids == NEURON, :], spike_counts[cluster_ids == NEURON, :]))

roc_auc, cluster_ids = roc_single_event(spikes[PROBE].times, spikes[PROBE].clusters,
                                        opto_train_times, pre_time=PRE_TIME, post_time=POST_TIME)
mod_index = 2 * (roc_auc - 0.5)

# Get region
region = remap(clusters[PROBE].atlas_id[NEURON])[0]

print(f'Area under ROC curve: {roc_auc[cluster_ids == NEURON][0]:.2f}')
print(f'Modulation index: {mod_index[cluster_ids == NEURON][0]:.2f}')

# %% Plot PSTH
colors, dpi = figure_style()
p, (ax, ax_roc) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
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
ax.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)',
       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3),
       ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax_roc.plot([0, 1], [0, 1], ls='--', color='grey')
ax_roc.plot(fpr, tpr, color='orange', lw=1.5)
ax_roc.set(ylabel='True positive rate', xlabel='False positive rate',
           xticks=[0, .5, 1], yticks=[0, .5, 1])
sns.despine(trim=True, ax=ax_roc)

plt.tight_layout()

plt.savefig(join(fig_path, f'{region}_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}'), dpi=600)
plt.savefig(join(fig_path, f'{region}_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))



