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
BIN_SIZE = 0.01
SMOOTHING = 0.05
BASELINE = [-1, 0]
MIN_FR = 0.1
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive')

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
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Take slice of dataframe
    these_neurons = light_neurons[(light_neurons['modulated'] == 1)
                                  & (light_neurons['eid'] == eid)
                                  & (light_neurons['probe'] == probe)]

    # Get peri-event time histogram
    peths, _ = calculate_peths(spikes.times, spikes.clusters,
                               these_neurons['neuron_id'].values,
                               opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
    tscale = peths['tscale']

    # Loop over neurons
    for n, index in enumerate(these_neurons.index.values):
        if np.mean(peths['means'][n, :]) > MIN_FR:
            # Calculate percentage change in firing rate
            peth_perc = ((peths['means'][n, :]
                          - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                         / np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))])) * 100

            # Calculate percentage change in firing rate
            peth_ratio = ((peths['means'][n, :]
                           - np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))]))
                          / (peths['means'][n, :]
                             + np.mean(peths['means'][n, ((tscale > BASELINE[0]) & (tscale < BASELINE[1]))])))

            # Add to dataframe
            peths_df = pd.concat((peths_df, pd.DataFrame(index=[peths_df.shape[0]], data={
                'peth': [peths['means'][n, :]], 'peth_perc': [peth_perc], 'peth_ratio': [peth_ratio],
                'region': these_neurons.loc[index, 'full_region'], 'modulation': these_neurons.loc[index, 'mod_index_late'],
                'neuron_id': these_neurons.loc[index, 'neuron_id'], 'subject': these_neurons.loc[index, 'subject'],
                'eid': these_neurons.loc[index, 'eid'], 'acronym': these_neurons.loc[index, 'region']})))

# %% Plot

VMIN = -1
VMAX = 1

# Plot all neurons
peths_df = peths_df.sort_values('modulation', ascending=False)
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(4, 3), dpi=dpi)
img = ax1.imshow(np.array(peths_df['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1])
cbar = f.colorbar(img, ax=ax1, shrink=0.7)
cbar.ax.set_ylabel('Normalized change in firing rate', rotation=270, labelpad=10)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax1.set(xlabel='Time (s)', yticks=[], ylabel=f'Sig. modulated neurons (n={peths_df.shape[0]})')
ax1.plot([0, 0], [-1, 1], ls='--', color='k')

plt.tight_layout()
plt.savefig(join(fig_path, 'figure2_all_neurons.pdf'))

# %%
# Plot per region
peths_df = peths_df.sort_values(['region', 'modulation'], ascending=[True, False])
f, ((ax_hc, ax_snr, ax_ppc, ax_am, ax_mpfc, ax_th),
    (ax_pag, ax_pir, ax_orb, ax_bc, ax_sc, ax_str)) = plt.subplots(2, 6, figsize=(9, 4), dpi=dpi)
title_font = 7

these_peths = peths_df[peths_df['region'] == 'Medial prefrontal cortex']
img = ax_mpfc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_mpfc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_mpfc.set_title('Medial prefrontal cortex', fontsize=title_font)
ax_mpfc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Orbitofrontal cortex']
img = ax_orb.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_orb.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_orb.set_title('Orbitofrontal', fontsize=title_font)
ax_orb.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Amygdala']
img = ax_am.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_am.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_am.set_title('Amygdala', fontsize=title_font)
ax_am.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Posterior parietal cortex']
img = ax_ppc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_ppc.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_ppc.set_title('Posterior parietal cortex', fontsize=title_font)
ax_ppc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Hippocampus']
img = ax_hc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_hc.set(yticks=[1], yticklabels=[these_peths.shape[0]], ylabel='Mod. neurons')
ax_hc.set_title('Hippocampus', fontsize=title_font)
ax_hc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Piriform']
img = ax_pir.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_pir.set(xlabel='Time (s)', yticks=[1], yticklabels=[these_peths.shape[0]])
ax_pir.set_title('Piriform', fontsize=title_font)
ax_pir.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Thalamus']
img = ax_th.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_th.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_th.set_title('Thalamus', fontsize=title_font)
ax_th.plot([0, 0], [-1, 1], ls='--', color='k')


these_peths = peths_df[peths_df['region'] == 'Tail of the striatum']
img = ax_str.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_str.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_str.set_title('Tail of the striatum', fontsize=title_font)
ax_str.plot([0, 0], [-1, 1], ls='--', color='k')


these_peths = peths_df[peths_df['region'] == 'Barrel cortex']
img = ax_bc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_bc.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_bc.set_title('Barrel cortex', fontsize=title_font)
ax_bc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Periaqueductal gray']
img = ax_pag.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_pag.set(yticks=[1], yticklabels=[these_peths.shape[0]], ylabel='Mod. neurons', xlabel='Time (s)')
ax_pag.set_title('Periaqueductal gray', fontsize=title_font)
ax_pag.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Superior colliculus']
img = ax_sc.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_sc.set(yticks=[1], yticklabels=[these_peths.shape[0]], xlabel='Time (s)')
ax_sc.set_title('Superior colliculus', fontsize=title_font)
ax_sc.plot([0, 0], [-1, 1], ls='--', color='k')

these_peths = peths_df[peths_df['region'] == 'Substantia nigra']
img = ax_snr.imshow(np.array(these_peths['peth_ratio'].tolist()), cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 vmin=VMIN, vmax=VMAX, extent=[-T_BEFORE, T_AFTER, -1, 1], interpolation='none')
ax_snr.set(yticks=[1], yticklabels=[these_peths.shape[0]])
ax_snr.set_title('Substantia nigra', fontsize=title_font)
ax_snr.plot([0, 0], [-1, 1], ls='--', color='k')

plt.tight_layout()

"""
cb_ax = f.add_axes([0.75, 0.15, 0.01, 0.3])
cbar = f.colorbar(mappable=ax_mpfc.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Norm. change\nin firing rate', rotation=270, labelpad=18)
"""

#plt.tight_layout(pad=3)
plt.savefig(join(fig_path, 'figure2_heatmap_per_region.pdf'))

