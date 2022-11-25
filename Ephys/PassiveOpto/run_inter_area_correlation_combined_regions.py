# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:23:17 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, get_neuron_qc,
                                 calculate_peths, combine_regions, load_subjects)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
T_BEFORE = 1
T_AFTER = 3
BIN_SIZE = 0.2
SMOOTHING = 0
SUBTRACT_MEAN = False
OVERWRITE = True

# Query sessions
rec = query_ephys_sessions(one=one)

# Initialize
fig_path, save_path = paths()
subjects = load_subjects()
artifact_neurons = get_artifact_neurons()

if OVERWRITE:
    corr_df = pd.DataFrame()
else:
    corr_df = pd.read_csv(join(save_path, f'combined_region_corr_{int(BIN_SIZE*1000)}.csv'))
    rec = rec[~np.isin(rec['eid'], corr_df['eid'])]

for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting {subject}, {date} [{i+1} of {len(np.unique(rec["eid"]))}]')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        continue

    # Load in neural data of both probes
    region_spikes = dict()
    for k, (pid, probe) in enumerate(zip(rec.loc[rec['eid'] == eid, 'pid'].values,
                                         rec.loc[rec['eid'] == eid, 'probe'].values)):

        try:
            sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
        except Exception as err:
            print(err)
            continue

        # Filter neurons that pass QC and artifact neurons
        qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters['region'] = combine_regions(remap(clusters['acronym']))

        # Get binned spike counts per region
        for j, region in enumerate(np.unique(clusters['region'])):
            if region == 'root':
                continue

            # Get spike counts
            peths, spike_counts = calculate_peths(
                spikes.times, spikes.clusters,
                clusters['cluster_id'][(clusters['region'] == region)
                                       & np.isin(clusters['cluster_id'], clusters_pass)],
                opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)

            if SUBTRACT_MEAN:
                # Subtract mean PSTH from each opto stim
                for tt in range(spike_counts.shape[0]):
                    spike_counts[tt, :, :] = spike_counts[tt, :, :] - peths['means']

            # Add to dictionary
            if (k == 1) & (region in region_spikes.keys()):
                # Concatenate activity from both probes that recorded the same region
                region_spikes[region] = np.concatenate((region_spikes[region], spike_counts), axis=1)
            else:
                region_spikes[region] = spike_counts

            # Get time scale
            tscale = peths['tscale']

    # Get pairwise neural correlations between all neuron pairs in both regions
    these_regions = list(region_spikes.keys())
    for r1, region_1 in enumerate(these_regions[:-1]):
        for r2, region_2 in enumerate(these_regions[r1+1:]):
            corr_mean = np.empty(tscale.shape)
            for tt in range(tscale.shape[0]):  # Loop over time bins
                pairwise_corr = []
                for n1 in range(region_spikes[region_1].shape[1]):  # Neurons in region 1
                    for n2 in range(region_spikes[region_2].shape[1]):  # Neurons in region 2
                        r, _ = pearsonr(region_spikes[region_1][:, n1, tt],
                                        region_spikes[region_2][:, n2, tt])
                        pairwise_corr.append(r)
                corr_mean[tt] = np.nanmean(pairwise_corr)

                # Baseline subtract
                corr_time_bl = corr_mean - np.mean(corr_mean[tscale < 0])

            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'r': corr_mean, 'r_baseline': corr_time_bl, 'time': tscale, 'region_1': region_1,
                'region_2': region_2, 'region_pair': f'{region_1}-{region_2}', 'subject': subject,
                'n_region_1': region_spikes[region_1].shape[1], 'n_region_2': region_spikes[region_2].shape[1],
                'eid': eid, 'date': date, 'sert-cre': sert_cre})), ignore_index=True)

    # Save to disk
    corr_df.to_csv(join(save_path, f'combined_region_corr_{int(BIN_SIZE*1000)}.csv'))