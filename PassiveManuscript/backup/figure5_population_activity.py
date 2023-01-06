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
from scipy.stats import kruskal
from brainbox.metrics.single_units import spike_sorting_metrics
from matplotlib.patches import Rectangle
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import (paths, load_passive_opto_times, combine_regions, load_subjects,
                                 get_artifact_neurons, remap, get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
REGIONS = ['M2', 'OFC', 'mPFC']
T_BEFORE = 1  # for plotting
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.1
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in light modulated neurons
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False, abbreviate=True)
light_neurons = light_neurons[light_neurons['full_region'] != 'root']

# Load in light artifact neurons
artifact_neurons = get_artifact_neurons()

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
    #these_neurons = light_neurons[(light_neurons['modulated'] == 1) & (light_neurons['pid'] == pid)]
    these_neurons = light_neurons[light_neurons['pid'] == pid]

    # Get session details
    eid = np.unique(these_neurons['eid'])[0]
    probe = np.unique(these_neurons['probe'])[0]
    subject = np.unique(these_neurons['subject'])[0]
    date = np.unique(these_neurons['date'])[0]
    print(f'Starting {subject}, {date}, {probe} ({i+1} of {len(np.unique(light_neurons["pid"]))})')

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

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    # Exclude artifact neurons
    clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values])
    if clusters_pass.shape[0] == 0:
            continue

    # Select QC pass neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes.clusters))]

    # Get regions from Beryl atlas
    clusters['region'] = remap(clusters['acronym'], combine=True)
    clusters_regions = clusters['region'][clusters_pass]

    # Get peri-event time histogram
    peths, _ = calculate_peths(spikes.times, spikes.clusters, clusters_pass,
                               opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
    tscale = peths['tscale']

    # Loop over regions
    for j, reg in enumerate(REGIONS):
        if np.sum(clusters_regions == reg) == 0:
            continue

        # Get population activity
        pop_act = peths['means'][clusters_regions == reg]

        # Normalize and offset mean for plotting
        pop_mean = pop_act.mean(axis=0)
        pop_mean_bl = pop_mean - np.mean(pop_mean[tscale < 0])
        pop_median = np.median(pop_act, axis=0)
        pop_median_bl = pop_mean - np.median(pop_mean[tscale < 0])
        pop_var = np.std(pop_act, axis=0)
        pop_var_bl = pop_var - np.mean(pop_var[tscale < 0])
        pop_ff = np.std(pop_act, axis=0) / np.mean(pop_act, axis=0)
        pop_ff_bl = pop_ff - np.mean(pop_ff[tscale < 0])

        peths_df = pd.concat((peths_df, pd.DataFrame(data={
            'mean': pop_mean, 'median': pop_median, 'var': pop_var, 'fano': pop_ff,
            'mean_bl': pop_mean_bl, 'median_bl': pop_median_bl, 'var_bl': pop_var_bl, 'fano_bl': pop_ff_bl,
            'time': peths['tscale'], 'region': reg, 'subject': subject, 'date': date, 'pid': pid})),
            ignore_index=True)


# Do statistics
mean_table_df = peths_df.pivot(index='time', columns=['region', 'pid'], values='mean_bl')
mean_table_df = mean_table_df.reset_index()
for i in mean_table_df.index.values:
    mean_table_df.loc[i, 'p_value'] = kruskal(mean_table_df.loc[i, 'M2'], mean_table_df.loc[i, 'mPFC'],
                                              mean_table_df.loc[i, 'OFC'])[1]

var_table_df = peths_df.pivot(index='time', columns=['region', 'pid'], values='var_bl')
var_table_df = var_table_df.reset_index()
for i in var_table_df.index.values:
    var_table_df.loc[i, 'p_value'] = kruskal(var_table_df.loc[i, 'M2'], var_table_df.loc[i, 'mPFC'],
                                             var_table_df.loc[i, 'OFC'])[1]

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)

ax1.add_patch(Rectangle((0, -4), 1, 6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='mean_bl', data=peths_df, ax=ax1, hue='region', errorbar='se',
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
ax1.set(xlabel='Time (s)', ylabel='Population activity (spks/s)',
        ylim=[-1, 1], xticks=[-1, 0, 1, 2])
leg = ax1.legend(frameon=True, prop={'size': 5.5}, loc='lower left')
leg.get_frame().set_linewidth(0.0)

ax2.add_patch(Rectangle((0, -2), 1, 4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='var_bl', data=peths_df, ax=ax2, hue='region', errorbar='se',
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
ax2.set(xlabel='Time (s)', ylabel='Population variance (std)',
        xticks=[-1, 0, 1, 2], ylim=[-2, 2.05])
#ax2.plot(var_table_df.loc[var_table_df['p_value'] < 0.05, 'time'],
#         np.ones(np.sum(var_table_df['p_value'] < 0.05))*2, color='k')
leg = ax2.legend(frameon=True, prop={'size': 5.5}, loc='lower left')
leg.get_frame().set_linewidth(0.0)

ax3.add_patch(Rectangle((0, -2), 1, 4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='fano', data=peths_df, ax=ax3, hue='region', errorbar='se',
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
ax3.set(xlabel='Time (s)', ylabel='Fano factor',
        xticks=[-1, 0, 1, 2], ylim=[0.5, 2])
#ax2.plot(var_table_df.loc[var_table_df['p_value'] < 0.05, 'time'],
#         np.ones(np.sum(var_table_df['p_value'] < 0.05))*2, color='k')
leg = ax3.legend(frameon=True, prop={'size': 5.5}, loc='lower left')
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'population_activity.pdf'))







