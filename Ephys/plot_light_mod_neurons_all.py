#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:53:54 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
from serotonin_functions import paths, figure_style, get_full_region_name, load_trials, remap, load_opto_times
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from oneibl.one import ONE
one = ONE()

# Settings
T_BEFORE = 1
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.025

# Paths
_, fig_path, save_path = paths()
fig_path_light = join(fig_path, '5HT', 'light-modulated-neurons')
save_path = join(save_path, '5HT')
fig_path = join(fig_path, '5HT', 'light-mod-neurons-overview-per-region')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Drop root
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'root' in j])

peth_df = pd.DataFrame()
for i, eid in enumerate(np.unique(light_neurons['eid'])):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    # Load in laser pulses
    opto_times = load_opto_times(eid, one=one)

    for p, probe in enumerate(spikes.keys()):

        # Light modulated neurons
        mod_neurons = light_neurons.loc[(light_neurons['modulated'] == True) & (light_neurons['eid'] == eid)
                                       & (light_neurons['probe'] == probe), 'cluster_id'].values
        peths, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters, mod_neurons,
                                   opto_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        roc_auc = light_neurons.loc[(light_neurons['modulated'] == True) & (light_neurons['eid'] == eid)
                                    & (light_neurons['probe'] == probe), 'roc_auc'].values
        regions = light_neurons.loc[(light_neurons['modulated'] == True) & (light_neurons['eid'] == eid)
                                          & (light_neurons['probe'] == probe), 'region'].values

        for k, neuron_id in enumerate(mod_neurons):
            peth_df = peth_df.append(pd.DataFrame(index=[peth_df.shape[0]+1], data={
                'eid': eid, 'subject': subject, 'date': date, 'probe': probe, 'neuron_id': neuron_id,
                'region': regions[k], 'roc_auc': roc_auc[k], 'peth': [peths.means[k,:]]}))

# %% Plot per region
peth_df['full_region'] = get_full_region_name(peth_df['region'])
peth_df = peth_df.sort_values(['full_region', 'roc_auc'], ascending=False)
for r, region in enumerate(np.unique(peth_df['full_region'])):
    all_peth = []
    for k, ind in enumerate(peth_df[peth_df['full_region'] == region].index):
        all_peth.append(peth_df.loc[ind, 'peth'] / peth_df.loc[ind, 'peth'].max())
    all_peth = np.vstack(all_peth)

    # Plot
    figure_style()
    f, ax1 = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    hm = sns.heatmap(all_peth)
    ax1.plot()
    ax1.set(title=region, xlabel='Time (s)', xticks=np.linspace(0, peths.tscale.shape[0], 7),
            xticklabels=np.linspace(-T_BEFORE, T_AFTER, 7))
    ax1.axes.yaxis.set_visible(False)
    hm.collections[0].colorbar.set_label('Norm. firing rate')
    plt.tight_layout()
    plt.savefig(join(fig_path, region))
    plt.close(f)
