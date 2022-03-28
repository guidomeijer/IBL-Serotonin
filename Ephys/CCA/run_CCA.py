#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1)

# Settings
NEURON_QC = True
MIN_NEURONS = 5  # per region
N_SPONT_WIN = 500  # number of time windows in spontaneous activity
WIN_SIZE = 0.1  # window size in seconds
EARLY_WIN_START = 0
LATE_WIN_START = 0.9
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Query sessions
rec = query_ephys_sessions(one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

cca_df = pd.DataFrame()
for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    print(f'Starting {subject}, {date}')

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

    # Load in neural data of both probes of the recording
    spikes, clusters, channels = dict(), dict(), dict()
    for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
        clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])

        # Filter neurons that pass QC and artifact neurons
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes.clusters)
        clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = remap(clusters[probe]['atlas_id'], combine=True)

    # Get time intervals for opto and spontaneous activity
    np.random.seed(42)
    spont_on_times = np.sort(np.random.uniform(opto_train_times[0] - (6 * 60), opto_train_times[0],
                                               size=N_SPONT_WIN))
    spont_times = np.column_stack((spont_on_times, spont_on_times + WIN_SIZE))
    early_times = np.column_stack((opto_train_times + EARLY_WIN_START,
                                   opto_train_times + EARLY_WIN_START + WIN_SIZE))
    late_times = np.column_stack((opto_train_times + LATE_WIN_START,
                                  opto_train_times + LATE_WIN_START + WIN_SIZE))

    # Create population activity arrays for all regions
    pop_spont, pop_early, pop_late = dict(), dict(), dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):

             # Select spikes and clusters in this brain region
             clusters_in_region = np.where(clusters[probe]['region'] == region)[0]
             spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)
                                               & np.isin(spikes[probe].clusters, clusters_pass)]
             clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)
                                                  & np.isin(spikes[probe].clusters, clusters_pass)]
             if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
                 pop_spont[region], _ = get_spike_counts_in_bins(spks_region, clus_region, spont_times)
                 pop_spont[region] = pop_spont[region].T
                 pop_early[region], _ = get_spike_counts_in_bins(spks_region, clus_region, early_times)
                 pop_early[region] = pop_early[region].T
                 pop_late[region], _ = get_spike_counts_in_bins(spks_region, clus_region, late_times)
                 pop_late[region] = pop_late[region].T

    # Perform CCA per region pair
    for region_1 in pop_spont.keys():
        for region_2 in pop_spont.keys():
            if region_1 == region_2:
                continue

            # Calculate population correlation during spontaneous activity
            x_train, x_test, y_train, y_test = train_test_split(
                pop_spont[region_1], pop_spont[region_2], test_size=0.5, random_state=42, shuffle=True)
            cca.fit(x_train, y_train)
            spont_x, spont_y = cca.transform(x_test, y_test)
            r_spont, _ = pearsonr(np.squeeze(spont_x), np.squeeze(spont_y))

            # Use fitted CCA axis to get population correlation during stimulus windows
            early_x, early_y = cca.transform(pop_early[region_1], pop_early[region_2])
            r_early, _ = pearsonr(np.squeeze(early_x), np.squeeze(early_y))
            late_x, late_y = cca.transform(pop_late[region_1], pop_late[region_2])
            r_late, _ = pearsonr(np.squeeze(late_x), np.squeeze(late_y))

            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]+1], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'r_spont': r_spont, 'r_early': r_early, 'r_late': r_late})))

# Save results
cca_df.to_csv(join(save_path, 'cca_results.csv'))






