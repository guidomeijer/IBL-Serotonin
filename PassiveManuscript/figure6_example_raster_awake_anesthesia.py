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
from matplotlib.patches import Rectangle
from os.path import join
from serotonin_functions import (load_passive_opto_times, paths, get_artifact_neurons,
                                 query_ephys_sessions, figure_style, get_neuron_qc)
from brainbox.processing import bincount2D
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure6')

# Recording
SUBJECT = 'ZFM-04820'
DATE = '2022-09-16'
PROBE = 'probe00'

# Query sessions
rec = query_ephys_sessions(anesthesia=True, one=one)
pid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'pid'].values[0]
eid = rec.loc[(rec['subject'] == SUBJECT) & (rec['date'] == DATE) & (rec['probe'] == PROBE), 'eid'].values[0]

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Load in opto times
opto_times_awake, _ = load_passive_opto_times(eid, one=one)
opto_times_anes, _ = load_passive_opto_times(eid, anesthesia=True, one=one)

# Load in spikes
spikes_times = one.load_dataset(eid, 'spikes.times.npy', collection=f'alf/{PROBE}/pykilosort')
spikes_clusters = one.load_dataset(eid, 'spikes.clusters.npy', collection=f'alf/{PROBE}/pykilosort')
spikes_depths = one.load_dataset(eid, 'spikes.depths.npy', collection=f'alf/{PROBE}/pykilosort')

qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes_times = spikes_times[np.isin(spikes_clusters, clusters_pass)]
spikes_depths = spikes_depths[np.isin(spikes_clusters, clusters_pass)]
spikes_clusters = spikes_clusters[np.isin(spikes_clusters, clusters_pass)]

# Convert to mm
spikes_depths = spikes_depths / 1000

# Get spike raster
R, times, depths = bincount2D(spikes_times, spikes_depths, xbin=0.01, ybin=0.02, weights=None)

# %% Plot figure
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 1.75), sharey=True, dpi=dpi)

for i in range(opto_times_anes.shape[0]):
    ax1.add_patch(Rectangle((opto_times_awake[i], 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[opto_times_awake[0] - 5, opto_times_awake[0] + 30], ylim=[0, 4], ylabel='Depth (mm)')
ax1.set_title('Awake', color=colors['awake'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 1, ax1.get_xlim()[0] + 6])
ax1.text(ax1.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

for i in range(opto_times_anes.shape[0]):
    ax2.add_patch(Rectangle((opto_times_anes[i], 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
ax2.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax2.set(xlim=[opto_times_anes[6] - 5, opto_times_anes[6] + 30], ylim=[0, 4])
ax2.set_title('Anesthesia', color=colors['anesthesia'], fontweight='bold')
ax2.set(xticks=[ax2.get_xlim()[0] + 1, ax2.get_xlim()[0] + 6])
ax2.text(ax2.get_xlim()[0] + 3.5, 4.4, '5s', ha='center', va='center')
ax2.axes.get_xaxis().set_visible(False)
ax2.invert_yaxis()

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'raster_awake_anesthesia.pdf'))
