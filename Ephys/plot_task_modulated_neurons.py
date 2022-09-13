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
from matplotlib.patches import Rectangle
import pandas as pd
from os import mkdir
import seaborn as sns
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from serotonin_functions import paths, load_trials, peri_multiple_events_time_histogram
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.025
PLOT_LATENCY = False
OVERWRITE = True
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'TaskNeurons')

# Load in data
all_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

for i, pid in enumerate(np.unique(all_neurons['pid'])):

    # Get eid
    eid = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(all_neurons.loc[all_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}, {probe}')

    # Load in trials
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Take slice of dataframe
    modulated = all_neurons[(all_neurons['pid'] == pid) & (all_neurons['opto_modulated'] == 1)]


    colors, dpi = figure_style()
    for n, ind in enumerate(modulated.index.values):
        region = modulated.loc[ind, 'region']
        subject = modulated.loc[ind, 'subject']
        date = modulated.loc[ind, 'date']
        neuron_id = modulated.loc[ind, 'neuron_id']

        # Plot PSTH
        p, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
        peri_multiple_events_time_histogram(
            spikes.times, spikes.clusters, trials['goCue_times'], trials['laser_stimulation'],
            neuron_id, t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
            pethline_kwargs=[{'color': colors['no-stim'], 'lw': 1}, {'color': colors['stim'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['no-stim'], 'alpha': 0.3}, {'color': colors['stim'], 'alpha': 0.3}],
            raster_kwargs=[{'color': colors['no-stim'], 'lw': 0.5}, {'color': colors['stim'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        ax.set(ylabel='Firing rate (spikes/s)', xlabel='Time from trial start (s)',
               yticks=np.linspace(0, np.ceil(ax.get_ylim()[1]), 3), xticks=[-1, 0, 1, 2])
        if np.round(ax.get_ylim()[1]) % 2 == 0:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        sns.despine(trim=False)
        plt.tight_layout()

        plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=600)
        plt.close(p)
