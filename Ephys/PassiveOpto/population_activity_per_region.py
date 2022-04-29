#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import paths, load_passive_opto_times, combine_regions, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.1
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys')

# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)
light_neurons = light_neurons[light_neurons['full_region'] != 'root']

# Only select neurons from sert-cre mice
subjects = load_subjects()
light_neurons = light_neurons[light_neurons['subject'].isin(
    subjects.loc[subjects['sert-cre'] == 1, 'subject'].values)]

# %% Loop over sessions
peths_df = pd.DataFrame()
for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Get session details
    eid = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'eid'])[0]
    probe = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'probe'])[0]
    subject = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'date'])[0]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        continue

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except:
        continue    

    # Take slice of dataframe
    these_neurons = light_neurons[(light_neurons['modulated'] == 1)
                                  & (light_neurons['eid'] == eid)
                                  & (light_neurons['probe'] == probe)]

    # Get peri-event time histogram
    peths, _ = calculate_peths(spikes.times, spikes.clusters,
                               these_neurons['neuron_id'].values,
                               opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
    tscale = peths['tscale']
    
    # Loop over regions
    for j, reg in enumerate(these_neurons['full_region'].unique()):
        peths_df = pd.concat((peths_df, pd.DataFrame(data={
            'mean': peths['means'][these_neurons['full_region'] == reg].mean(axis=0),
            'median': np.median(peths['means'][these_neurons['full_region'] == reg], axis=0),
            'std': peths['means'][these_neurons['full_region'] == reg].std(axis=0),
            'time': peths['tscale'], 'region': reg, 'subject': subject, 'date': date, 'pid': pid})))

peths_df['çv'] = peths_df['std'] / peths_df['mean'] 
peths_df = peths_df.reset_index(drop=True)

# %% Plot
colors, dpi = figure_style()
f, axs = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)
sns.lineplot(x='time', y='mean', hue='region', data=peths_df, ax=axs[0], ci=68, palette='tab10')
axs[0].set(xlabel='Time (s)', ylabel='Mean population activity (spks/s)')
axs[0].legend(frameon=False)

sns.lineplot(x='time', y='median', hue='region', data=peths_df, ax=axs[1], ci=68)

sns.lineplot(x='time', y='std', hue='region', data=peths_df, ax=axs[2], ci=68)

sns.lineplot(x='time', y='cv', hue='region', data=peths_df, ax=axs[3], ci=68)



