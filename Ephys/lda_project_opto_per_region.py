#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from os import mkdir
from ibllib.io import spikeglx
from brainbox.task.closed_loop import roc_single_event
from my_functions import figure_style
import brainbox.io.one as bbone
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from serotonin_functions import paths, remap, query_ephys_sessions, load_opto_times
from one.api import ONE
lda = LinearDiscriminantAnalysis()
one = ONE()

# Settings
MIN_NEURONS = 5  # per region
PLOT = False
T_BEFORE = 1
T_AFTER = 2
BIN_SIZE = 0.1
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'decoding_opto_pulses')
save_path = join(save_path, '5HT')

# Query sessions
eids, _ = query_ephys_sessions(one=one)

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

lda_dist_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times = load_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):

        # Select spikes of passive period
        start_passive = opto_train_times[0] - 360
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > start_passive]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > start_passive]

        # Filter neurons that pass QC
        if 'metrics' in clusters[probe].keys():
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        else:
            print('No neuron QC, using all units')
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]

        # Get a number of random onset times in the spontaneous activity as control
        control_times = np.random.uniform(low=start_passive, high=opto_train_times[0],
                                          size=opto_train_times.shape[0], )
        all_times = np.concatenate((control_times, opto_train_times))
        laser_on = np.concatenate((np.zeros(control_times.shape[0]), np.ones(control_times.shape[0])))

        # Loop over regions
        for r, region in enumerate(np.unique(clusters[probe]['acronym'])):
            print(f'Run LDA on region {region}')

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters[probe].metrics.cluster_id[clusters[probe]['acronym'] == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            lda_dist = np.empty(BIN_CENTERS.shape[0])
            for b, bin_center in enumerate(BIN_CENTERS):
                times = np.column_stack((((all_times + bin_center) - (BIN_SIZE / 2)),
                                         ((all_times + bin_center) + (BIN_SIZE / 2))))
                pop_vector, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
                pop_vector = pop_vector.T
                lda_projection = lda.fit_transform(pop_vector, laser_on)
                lda_dist[b] = (np.abs(np.mean(lda_projection[laser_on == 0]))
                               + np.abs(np.mean(lda_projection[laser_on == 1])))

            plt.plot(BIN_CENTERS, lda_dist)


