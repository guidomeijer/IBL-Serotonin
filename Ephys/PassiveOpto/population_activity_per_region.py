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
REGIONS = ['M2', 'mPFC', 'ORB', 'Amyg', 'Thal', 'Hipp', 'PPC', 'PAG', 'Pir']
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.1
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys')

# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False, abbreviate=True)
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
        # Normalize and offset mean for plotting
        pop_mean = peths['means'][these_neurons['full_region'] == reg].mean(axis=0)
        pop_mean = (pop_mean / np.max(pop_mean))
        pop_mean = pop_mean - np.mean(pop_mean[tscale < 0])
        pop_median = np.median(peths['means'][these_neurons['full_region'] == reg], axis=0)
        pop_median = (pop_median / np.max(pop_median))
        pop_median = pop_median - np.median(pop_median[tscale < 0])
        
        peths_df = pd.concat((peths_df, pd.DataFrame(data={
            'mean': pop_mean, 'median': pop_median,
            'std': peths['means'][these_neurons['full_region'] == reg].std(axis=0),
            'time': peths['tscale'], 'region': reg, 'subject': subject, 'date': date, 'pid': pid})),
            ignore_index=True)
    

# %% Plot

colors, dpi = figure_style()
peths_df = peths_df[peths_df['region'].isin(REGIONS)]
ORDER = ['Hipp', 'PPC', 'M2', 'Thal', 'Amyg', 'Pir', 'ORB', 'mPFC', 'PAG']
COLORS = [colors[i] for i in ORDER]

f, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=dpi)

offset = np.arange(len(peths_df['region'].unique())) / 5
peths_df['mean_offset'] = peths_df['mean'].copy()
peths_df['median_offset'] = peths_df['median'].copy()
for k, reg in enumerate(ORDER):
    peths_df.loc[peths_df['region'] == reg, 'mean_offset'] = (
        peths_df.loc[peths_df['region'] == reg, 'mean'] + offset[k])
    peths_df.loc[peths_df['region'] == reg, 'median_offset'] = (
        peths_df.loc[peths_df['region'] == reg, 'median'] + offset[k])
sns.lineplot(x='time', y='mean_offset', hue='region', data=peths_df, ax=ax, ci=68, hue_order=ORDER,
             palette=COLORS, legend=None)
ax.axis('off')
for k, y_text in enumerate(offset):
    ax.text(-1, y_text, ORDER[k], ha='right', va='top', color=COLORS[k])
ax.plot([0, 0], [-0.5, np.max(offset)+0.1], color='grey', ls='--')
ax.plot([-1, -0.5], [-0.3, -0.3], color='k')    
ax.text(-0.75, -0.42, '0.5s', ha='center')

sns.despine(trim=True)



