#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir, exists
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from os import mkdir
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, remove_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.05
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'LightModNeurons')

# Load in data
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

for i, pid in enumerate(np.unique(all_neurons['pid'])):

    # Get eid
    eid = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}, {probe}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Take slice of dataframe
    modulated = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['modulated'] == 1)]

    for n, ind in enumerate(modulated.index.values):
        region = modulated.loc[ind, 'region']
        subject = modulated.loc[ind, 'subject']
        date = modulated.loc[ind, 'date']
        neuron_id = modulated.loc[ind, 'neuron_id']
        if exists(join(fig_path, 'Regions', region,
                       f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}.png')):
            continue

        # Plot PSTH
        colors, dpi = figure_style()
        p, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        peri_event_time_histogram(spikes.times, spikes.clusters, opto_train_times,
                                  neuron_id, t_before=T_BEFORE, t_after=T_AFTER,
                                  bin_size=BIN_SIZE, include_raster=True, error_bars='sem', ax=ax,
                                  pethline_kwargs={'color': 'black', 'lw': 1},
                                  errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                  raster_kwargs={'color': 'black', 'lw': 0.3},
                                  eventline_kwargs={'lw': 0})
        ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
        """
        ax.set(ylabel='spikes/s', xlabel='Time (s)',
               title=f'Modulation index: {roc_auc[cluster_ids == cluster][0]:.2f}',
               yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        """
        ax.set(ylabel='Firing rate (spks/s)', xlabel='Time (s)',
               yticks=[np.round(ax.get_ylim()[1])],
               ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
        ax.plot([0, 1], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                         ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=2, color='royalblue')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.tight_layout()

        if not isdir(join(fig_path, 'Regions', f'{region}')):
            mkdir(join(fig_path, 'Regions', f'{region}'))
        plt.savefig(join(fig_path, 'Regions', region,
                         f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}'), dpi=600)
        if not isdir(join(fig_path, 'Recordings', f'{subject}_{date}')):
            mkdir(join(fig_path, 'Recordings', f'{subject}_{date}'))
        plt.savefig(join(fig_path, 'Recordings', f'{subject}_{date}',
                         f'{subject}_{date}_{probe}_neuron{neuron_id}_{region}'), dpi=600)
        plt.close(p)
