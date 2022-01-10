#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from brainbox.population.decode import get_spike_counts_in_bins
import pandas as pd
from os import mkdir
from brainbox.task.closed_loop import roc_single_event
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_ephys_sessions, load_passive_opto_times, get_artifact_neurons
from one.api import ONE
from ibllib.atlas import AllenAtlas, BrainRegions
ba = AllenAtlas()
br = BrainRegions()
one = ONE()

# Settings
MIN_NEURONS = 10
MIN_FR = 0.01  # spks/s
BASELINE = [-1, -0.2]
PLOT = True
OVERWRITE = True
NEURON_QC = True
BIN_CENTERS = np.arange(-0.9, 1.91, 0.2)
BIN_SIZE = 0.2
_, _, save_path = paths()

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

# Get artifact neurons
artifact_neurons = get_artifact_neurons()

pop_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

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

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Exclude artifact neurons
        clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
            (artifact_neurons['eid'] == eid) & (artifact_neurons['probe'] == probe), 'neuron_id'].values])
        if clusters_pass.shape[0] == 0:
            continue

        # Get regions from Beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'], combine=True, brainregions=br)
        clusters_regions = clusters[probe]['acronym'][clusters_pass]

        # Loop over regions
        for r, region in enumerate(np.unique(clusters_regions)):
            if region == 'root':
                continue
            print(region)

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            # Loop over time bins
            pop_means, pop_stds = np.empty(BIN_CENTERS.shape), np.empty(BIN_CENTERS.shape)
            sparsity, noise_corr = np.empty(BIN_CENTERS.shape), np.empty(BIN_CENTERS.shape)
            for b, bin_center in enumerate(BIN_CENTERS):
                times = np.column_stack((((opto_train_times + bin_center) - (BIN_SIZE / 2)),
                                         ((opto_train_times + bin_center) + (BIN_SIZE / 2))))
                pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
                pop_vector = pop_vector.T
                
                # Get mean and variance of distribution of log-firing rates
                firing_rate = np.mean(pop_vector, axis=0) / BIN_SIZE
                log_rate = np.log10(firing_rate[firing_rate > MIN_FR])
                pop_means[b] = np.mean(log_rate)
                pop_stds[b] = np.std(log_rate)   
                
                # Calculate population sparsity
                sparsity[b] = ((1 - (1 / pop_vector.shape[1])
                                * ((np.sum(np.mean(pop_vector, axis=0))**2)
                                   / (np.sum(np.mean(pop_vector, axis=0)**2))))
                               / (1 - (1 / pop_vector.shape[1])))

                # Calculate noise correlations (remove neurons that do not spike at all)
                if pop_vector[:, np.sum(pop_vector, axis=0) > 0].shape[1] > 1:
                    corr_arr = np.corrcoef(pop_vector[:, np.sum(pop_vector, axis=0) > 0], rowvar=False)
                    corr_arr = corr_arr[np.triu_indices(corr_arr.shape[0])]
                    corr_arr = corr_arr[corr_arr != 1]
                    noise_corr[b] = np.mean(corr_arr)
                else:
                    noise_corr[b] = np.nan
            
            # Do baseline subtraction
            pop_means_bl = pop_means - np.mean(pop_means[(BIN_CENTERS > BASELINE[0]) & (BIN_CENTERS < BASELINE[1])])
            pop_stds_bl = pop_stds - np.mean(pop_stds[(BIN_CENTERS > BASELINE[0]) & (BIN_CENTERS < BASELINE[1])])
            noise_corr_bl = noise_corr - np.mean(noise_corr[(BIN_CENTERS > BASELINE[0]) & (BIN_CENTERS < BASELINE[1])])
            sparsity_bl = sparsity - np.mean(sparsity[(BIN_CENTERS > BASELINE[0]) & (BIN_CENTERS < BASELINE[1])])
            
            # Add to dataframe
            pop_df = pop_df.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'probe': probe, 'eid': eid,
                'mean': pop_means, 'var': pop_stds, 'mean_bl': pop_means_bl, 'std_bl': pop_stds_bl,
                'sparsity': sparsity, 'sparsity_bl': sparsity_bl, 'noise_corr': noise_corr, 
                'noise_corr_bl': noise_corr_bl, 'region': region, 'time': BIN_CENTERS}))

# Save output
pop_df.to_csv(join(save_path, 'population_metrics.csv'), index=False)
