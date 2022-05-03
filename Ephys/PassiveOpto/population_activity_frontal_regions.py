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
from matplotlib.patches import Rectangle
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import paths, load_passive_opto_times, combine_regions, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
REGIONS = ['M2', 'ORB', 'mPFC']
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

# Only select neurons from target regions
light_neurons = light_neurons[light_neurons['full_region'].isin(REGIONS)]

# %% Loop over sessions
peths_df = pd.DataFrame()
for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Take slice of dataframe
    these_neurons = light_neurons[(light_neurons['modulated'] == 1) & (light_neurons['pid'] == pid)] 

    # Get session details
    eid = np.unique(these_neurons['eid'])[0]
    probe = np.unique(these_neurons['probe'])[0]
    subject = np.unique(these_neurons['subject'])[0]
    date = np.unique(these_neurons['date'])[0]
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

    # Get peri-event time histogram
    peths, _ = calculate_peths(spikes.times, spikes.clusters,
                               these_neurons['neuron_id'].values,
                               opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
    tscale = peths['tscale']
    
    # Loop over regions
    for j, reg in enumerate(REGIONS):
        if np.sum(these_neurons['full_region'] == reg) == 0:
            continue
        # Normalize and offset mean for plotting
        pop_mean = peths['means'][these_neurons['full_region'] == reg].mean(axis=0)
        pop_mean_bl = pop_mean - np.mean(pop_mean[tscale < 0])
        pop_median = np.median(peths['means'][these_neurons['full_region'] == reg], axis=0)
        pop_median_bl = pop_mean - np.median(pop_mean[tscale < 0])
        pop_var = np.std(peths['means'][these_neurons['full_region'] == reg], axis=0)
        pop_var_bl = pop_var - np.mean(pop_var[tscale < 0])
        pop_cv = (np.std(peths['means'][these_neurons['full_region'] == reg], axis=0)
                  / np.mean(peths['means'][these_neurons['full_region'] == reg], axis=0))
        pop_cv_bl = pop_cv - np.mean(pop_cv[tscale < 0])
        
        peths_df = pd.concat((peths_df, pd.DataFrame(data={
            'mean': pop_mean, 'median': pop_median, 'var': pop_var, 'cv': pop_cv,
            'mean_bl': pop_mean_bl, 'median_bl': pop_median_bl, 'var_bl': pop_var_bl, 'cv_bl': pop_cv_bl,
            'time': peths['tscale'], 'region': reg, 'subject': subject, 'date': date, 'pid': pid})),
            ignore_index=True)
    
# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -4), 1, 6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='mean_bl', data=peths_df, ax=ax1, hue='region', ci=68,
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
ax1.set(xlabel='Time (s)', ylabel='Population activity (spks/s)',
        ylim=[-4, 2], xticks=[-1, 0, 1, 2])
leg = ax1.legend(frameon=True, prop={'size': 6})
leg.get_frame().set_linewidth(0.0)

ax2.add_patch(Rectangle((0, -0.3), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='cv_bl', data=peths_df, ax=ax2, hue='region', ci=68,
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS], legend=None)
ax2.set(xlabel='Time (s)', ylabel='Population variance (C.V.)',
        xticks=[-1, 0, 1, 2], ylim=[-0.3, 0.3], yticks=[-.3, -.2, -.1, 0, .1, .2, .3])
leg = ax2.legend(frameon=True, prop={'size': 6}, loc='lower left')
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)






