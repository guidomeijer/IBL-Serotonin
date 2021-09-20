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
one = ONE()

# Settings
PLOT = False
ITERATIONS = 10
T_BEFORE = 0.5
T_AFTER = 1.5
BIN_SIZE = 0.2
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

# Set paths
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'Ephys', 'LDA')
save_path = join(save_path, '5HT')

# Query sessions
eids, _ = query_ephys_sessions(selection='all', one=one)

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
        if spikes[probe] is None:
            continue

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


        print('LDA projection light pulses..')
        lda = LinearDiscriminantAnalysis()
        lda_dist = np.empty((ITERATIONS, BIN_CENTERS.shape[0]))
        for k in range(ITERATIONS):

            # Get a number of random onset times in the spontaneous activity as control
            control_times = np.random.uniform(low=start_passive, high=opto_train_times[0],
                                              size=opto_train_times.shape[0], )
            all_times = np.concatenate((control_times, opto_train_times))
            laser_on = np.concatenate((np.zeros(control_times.shape[0]), np.ones(opto_train_times.shape[0])))

            for b, bin_center in enumerate(BIN_CENTERS):
                times = np.column_stack((((all_times + bin_center) - (BIN_SIZE / 2)),
                                         ((all_times + bin_center) + (BIN_SIZE / 2))))
                pop_vector, cluster_ids = get_spike_counts_in_bins(spikes[probe].times,
                                                                   spikes[probe].clusters, times)
                pop_vector = pop_vector.T
                lda_projection = lda.fit_transform(pop_vector, laser_on)
                lda_dist[k, b] = (np.abs(np.mean(lda_projection[laser_on == 0]))
                                  + np.abs(np.mean(lda_projection[laser_on == 1])))

        plt.errorbar(BIN_CENTERS, np.mean(lda_dist, axis=0), yerr=np.std(lda_dist, axis=0))
        asd

