# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:06 2022

@author: Guido
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import paths, load_passive_opto_times
from brainbox.population.decode import get_spike_counts_in_bins
from sklearn.metrics import roc_auc_score
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = False
BASELINE = [0.5, 0]
PRE_TIME = 1
POST_TIME = 5
BIN_SIZE = 0.2
win_centers = np.arange(-PRE_TIME + (BIN_SIZE/2), POST_TIME, BIN_SIZE)

# Load in results
fig_path, save_path = paths()
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

if OVERWRITE:
    mod_idx_df = pd.DataFrame()
else:
    mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))
light_neurons = light_neurons[~np.isin(light_neurons['pid'], mod_idx_df['pid'])]

for i, pid in enumerate(np.unique(light_neurons['pid'])):

    # Get session data
    eid = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'eid'])[0]
    subject = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'subject'])[0]
    date = np.unique(light_neurons.loc[light_neurons['pid'] == pid, 'date'])[0]
    print(f'Processing {subject} {date}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Select only modulated neurons
    these_neurons = light_neurons[(light_neurons['modulated'] == 1) & (light_neurons['pid'] == pid)]
    spike_times = spikes.times[np.isin(spikes.clusters, these_neurons['neuron_id'])]
    spike_clusters = spikes.clusters[np.isin(spikes.clusters, these_neurons['neuron_id'])]
    if spike_times.shape[0] == 0:
        continue

    # Get spike counts for baseline windows
    baseline_wins = win_centers[(win_centers >= -BASELINE[0]) & (win_centers < BASELINE[1])]
    baseline_counts = np.empty((these_neurons.shape[0], opto_train_times.shape[0], baseline_wins.shape[0]))
    for itb, win_c in enumerate(baseline_wins):
        # Get spike counts for this
        times = np.column_stack(((opto_train_times + (win_c - (BIN_SIZE/2)),
                                  (opto_train_times + (win_c + (BIN_SIZE/2))))))
        baseline_counts[:,:,itb], _ = get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Get median spike count over all baseline windows
    baseline_median = np.median(baseline_counts, axis=2)

    # Loop over time bins
    roc_auc = np.empty((baseline_counts.shape[0], win_centers.shape[0]))
    for itb, win_c in enumerate(win_centers):

        # Get spike counts for this
        times = np.column_stack(((opto_train_times + (win_c - (BIN_SIZE/2)),
                                  (opto_train_times + (win_c + (BIN_SIZE/2))))))
        spike_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)

        # Loop over neurons
        for iin in range(spike_counts.shape[0]):

            # Calculate area under the ROC curve
            roc_auc[iin, itb] = roc_auc_score(np.concatenate((np.zeros(baseline_median.shape[1]),
                                                              np.ones(spike_counts.shape[1]))),
                                              np.concatenate((baseline_median[iin, :], spike_counts[iin, :])))

    # Rescale area under to curve to [-1, 1] range
    mod_idx = 2 * (roc_auc - 0.5)

    # Add to dataframe
    for iin, neuron_id in enumerate(these_neurons['neuron_id']):
        mod_idx_df = pd.concat((mod_idx_df, pd.DataFrame(index=[mod_idx_df.shape[0]+1], data={
            'pid': pid, 'subject': subject, 'date': date, 'neuron_id': neuron_id,
            'mod_idx': [mod_idx[iin, :]], 'time': [win_centers],
            'region': these_neurons.loc[these_neurons['neuron_id'] == neuron_id, 'region'].values[0]})))

    # Save output
    mod_idx_df.to_pickle(join(save_path, 'mod_over_time.pickle'))

