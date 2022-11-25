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
from matplotlib.patches import Rectangle
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from zetapy import getZeta
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, load_passive_opto_times
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
SUBJECT = 'ZFM-01802'
DATE = '2021-03-09'
PROBE = 'probe00'
NEURON = 349
# frontal cortex complex modulation
SUBJECT = 'ZFM-02600'
DATE = '2021-08-25'
PROBE = 'probe00'
NEURON = 442

SUBJECT = 'ZFM-01802'
DATE = '2021-03-11'
PROBE = 'probe00'
NEURON = 181
# good example complex modulation
SUBJECT = 'ZFM-01802'
DATE = '2021-03-11'
PROBE = 'probe00'
NEURON = 207
# Good example thalamus
SUBJECT = 'ZFM-01802'
DATE = '2021-03-09'
PROBE = 'probe00'
NEURON = 47
"""

# frontal cortex enhancement
SUBJECT = 'ZFM-03330'
DATE = '2022-02-15'
PROBE = 'probe00'
NEURON = 323

"""
# Good example CA1
SUBJECT = 'ZFM-01802'
DATE = '2021-03-09'
PROBE = 'probe00'
NEURON = 550
"""

T_BEFORE = 1  # for plotting
T_AFTER = 2
ZETA_BEFORE = 0  # baseline period to include for zeta test
PRE_TIME = [1, 0]  # for modulation index
POST_TIME = [0, 1]
BIN_SIZE = 0.05
SMOOTHING = 0.025
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure2')

# Get session details
ins = one.alyx.rest('insertions', 'list', date=DATE, subject=SUBJECT, name=PROBE)
pid = ins[0]['id']
eid = ins[0]['session']

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Select spikes of passive period
start_passive = opto_train_times[0] - 360
spikes.clusters = spikes.clusters[spikes.times > start_passive]
spikes.times = spikes.times[spikes.times > start_passive]

# Calculate ZETA
p_value, latencies, dZETA, dRate = getZeta(spikes.times[spikes.clusters == NEURON],
                                           opto_train_times - ZETA_BEFORE,
                                           intLatencyPeaks=4,
                                           tplRestrictRange=(0 + ZETA_BEFORE, 1 + ZETA_BEFORE),
                                           dblUseMaxDur=2 + ZETA_BEFORE,
                                           boolReturnZETA=True, boolReturnRate=True)
latency = latencies[3] - ZETA_BEFORE
dZETA['vecSpikeT'] = dZETA['vecSpikeT'] - ZETA_BEFORE

# Get spike counts for baseline and event timewindow
baseline_times = np.column_stack(((opto_train_times - PRE_TIME[0]), (opto_train_times - PRE_TIME[1])))
baseline_counts, cluster_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters,
                                                        baseline_times)
times = np.column_stack(((opto_train_times + POST_TIME[0]), (opto_train_times + POST_TIME[1])))
spike_counts, cluster_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, times)

fpr, tpr, _ = roc_curve(np.append(np.zeros(baseline_counts.shape[1]), np.ones(baseline_counts.shape[1])),
                        np.append(baseline_counts[cluster_ids == NEURON, :], spike_counts[cluster_ids == NEURON, :]))

roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                        opto_train_times, pre_time=PRE_TIME, post_time=POST_TIME)
mod_index = 2 * (roc_auc - 0.5)

# Get region
region = remap(clusters.acronym[NEURON])[0]

# Calculate mean spike rate
stim_intervals = np.vstack((opto_train_times, opto_train_times + 1)).T
spike_rate, neuron_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, stim_intervals)
stim_rate = spike_rate[neuron_ids == NEURON, :][0]
bl_intervals = np.vstack((opto_train_times - 1, opto_train_times)).T
spike_rate, neuron_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, bl_intervals)
bl_rate = spike_rate[neuron_ids == NEURON, :][0]

print(f'Area under ROC curve: {roc_auc[cluster_ids == NEURON][0]:.2f}')
print(f'Modulation index: {mod_index[cluster_ids == NEURON][0]:.2f}')
print(f'ZETA p-value: {p_value}')

# %% Plot PSTH
colors, dpi = figure_style()
p, ax = plt.subplots(1, 1, figsize=(1.75, 2), dpi=dpi)
ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                          NEURON, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                          smoothing=SMOOTHING,  include_raster=True, error_bars='sem', ax=ax,
                          pethline_kwargs={'color': 'black', 'lw': 1},
                          errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                          raster_kwargs={'color': 'black', 'lw': 0.3},
                          eventline_kwargs={'lw': 0})
ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
#ax.plot([0, 1], [0, 0], lw=2, color='royalblue')
ax.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)',
       yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3),
       ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_label_coords(-.2, .75)

plt.tight_layout()

plt.savefig(join(fig_path, f'{region}_{SUBJECT}_{DATE}_{PROBE}_neuron{NEURON}.pdf'))