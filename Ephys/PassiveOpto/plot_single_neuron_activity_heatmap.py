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
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from serotonin_functions import paths, load_passive_opto_times, combine_regions, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.01
SMOOTHING = 0.05
BASELINE = [-1, 0]
MIN_FR = 0.1
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons')

# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'])
light_neurons = light_neurons[light_neurons['full_region'] != 'root']

# Only select neurons from sert-cre mice
subjects = load_subjects()
light_neurons = light_neurons[light_neurons['subject'].isin(
    subjects.loc[subjects['sert-cre'] == 1, 'subject'].values)]

# %% Loop over sessions
peths_df = pd.DataFrame()
for i, eid in enumerate(np.unique(light_neurons['eid'])):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    
    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    # Loop over probes
    for p, probe in enumerate(spikes.keys()):
        
        # Take slice of dataframe
        these_neurons = light_neurons[(light_neurons['modulated'] == 1)
                                      & (light_neurons['eid'] == eid)
                                      & (light_neurons['probe'] == probe)]
        
        # Get peri-event time histogram
        peths, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters,
                                   these_neurons['neuron_id'].values,
                                   opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        tscale = peths['tscale']
        
        # Loop over neurons
        for n, index in enumerate(these_neurons.index.values):
            if np.mean(peths['means'][n, :]) > MIN_FR:
                # Calculate percentage change in firing rate
                peth_perc = ((peths['means'][n, :]
                              - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                             / (peths['means'][n, :]) + np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                
                # Add to dataframe
                peths_df = peths_df.append(pd.DataFrame(index=[peths_df.shape[0]], data={
                    'peth': [peths['means'][n, :]], 'peth_perc': [peth_perc],
                    'region': these_neurons.loc[index, 'full_region'], 'modulation': these_neurons.loc[index, 'roc_auc'],
                    'neuron_id': these_neurons.loc[index, 'neuron_id'], 'subject': these_neurons.loc[index, 'subject'],
                    'eid': these_neurons.loc[index, 'eid']}))
            
# %% Plot

VMIN = -100
VMAX = 100

# Plot all neurons
peths_df = peths_df.sort_values('modulation', ascending=False)
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=dpi)
img = ax1.imshow(np.array(peths_df['peth_perc'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True), 
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
cbar = f.colorbar(img, ax=ax1, shrink=0.7)
cbar.ax.set_ylabel('% change in firing rate', rotation=270)
ax1.set(xlabel='Time (s)', yticks=[], title=f'All significantly modulated neurons (n={peths_df.shape[0]})')
ax1.plot([0, 0], [-1, 1], ls='--', color='k')

# Plot per region
peths_df = peths_df.sort_values(['region', 'modulation'], ascending=[True, False])
f, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(8, 4), dpi=dpi)

these_peths = peths_df[peths_df['region'] == 'Amygdala']
img = ax1.imshow(np.array(these_peths['peth_perc'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True), 
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
ax1.set(xlabel='Time (s)', yticks=[], title=f'Amygdala (n={peths_df.shape[0]})')
ax1.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Hippocampus']
img = ax2.imshow(np.array(these_peths['peth_perc'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True), 
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
ax2.set(xlabel='Time (s)', yticks=[], title=f'Hippocampus (n={peths_df.shape[0]})')
ax2.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Orbitofrontal']
img = ax3.imshow(np.array(these_peths['peth_perc'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True), 
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
ax3.set(xlabel='Time (s)', yticks=[], title=f'Orbitofrontal (n={peths_df.shape[0]})')
ax3.plot([0, 0], [-1, 1], ls='--', color='k')



     
        
        