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
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (load_passive_opto_times, paths, get_artifact_neurons,
                                 query_ephys_sessions, figure_style, get_neuron_qc)
from brainbox.processing import bincount2D
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'Anesthesia', 'SpikeRasters')

# Query sessions
rec = query_ephys_sessions(anesthesia=True, one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    # Load in opto times
    opto_times_awake, _ = load_passive_opto_times(eid, one=one)
    opto_times_anes, _ = load_passive_opto_times(eid, anesthesia=True, one=one)

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    spikes_times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes_depths = spikes.depths[np.isin(spikes.clusters, clusters_pass)]
    spikes_clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

    # Get spike raster
    R, times, depths = bincount2D(spikes_times, spikes_depths, xbin=0.01, ybin=20, weights=None)

    # %% Plot figure
    colors, dpi = figure_style()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=600)

    ax1.add_patch(Rectangle((opto_times_awake[0], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax1.add_patch(Rectangle((opto_times_awake[1], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax1.add_patch(Rectangle((opto_times_awake[2], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax1.add_patch(Rectangle((opto_times_awake[3], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
               extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
    ax1.set(xlim=[opto_times_awake[0] - 5, opto_times_awake[0] + 25], ylim=[0, 4000], title='Awake',
            xlabel='Time since start of recording (s)', ylabel='Depth (um)')
    ax1.invert_yaxis()

    ax2.add_patch(Rectangle((opto_times_anes[0], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax2.add_patch(Rectangle((opto_times_anes[1], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax2.add_patch(Rectangle((opto_times_anes[2], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax2.add_patch(Rectangle((opto_times_anes[3], 0), 1, 4000, color='royalblue', alpha=0.25, lw=0))
    ax2.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
               extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
    ax2.set(xlim=[opto_times_anes[0] - 5, opto_times_anes[0] + 25], ylim=[0, 4000], title='Anesthesia',
            xlabel='Time since start of recording (s)')
    ax2.invert_yaxis()

    ch_depths = np.flip(channels['axial_um']) + 60
    for k in range(0, channels['acronym'].shape[0], 20):
        ax2.text(ax2.get_xlim()[1]+5, ch_depths[k], channels['acronym'][k], fontsize=5)

    plt.tight_layout()
    sns.despine(trim=True, offset=4)
    plt.savefig(join(fig_path, f'{subject}_{date}_{probe}.pdf'))
    plt.close(f)
