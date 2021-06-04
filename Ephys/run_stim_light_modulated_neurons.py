#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:03:53 2021
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
from brainbox.task.closed_loop import (roc_single_event, roc_between_two_events,
                                       generate_pseudo_stimuli, generate_pseudo_blocks)
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_sessions, load_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME = [0, 0.5]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'light-modulated-neurons')
save_path = join(save_path, '5HT')

# Query sessions
eids, _ = query_sessions(one=one)

stim_neurons = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    # Load trials dataframe
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):

        # Filter neurons that pass QC
        clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        cluster_regions = remap(clusters[probe].atlas_id[clusters_pass])

        # Determine stimulus responsive neurons
        print('Calculating stimulus responsive neurons..')
        roc_l_stim = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                      trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME)[0]
        roc_r_stim = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                      trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME)[0]
        roc_permut_l_stim = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_r_stim = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            _, contrast_l, contrast_r = generate_pseudo_stimuli(trials.shape[0], first5050=0)
            roc_permut_l_stim[k, :] = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                       trials.loc[contrast_l == 1, 'stimOn_times'],
                                                       pre_time=PRE_TIME, post_time=POST_TIME)[0]
            roc_permut_r_stim[k, :] = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                       trials.loc[contrast_r == 1, 'stimOn_times'],
                                                       pre_time=PRE_TIME, post_time=POST_TIME)[0]

        # Determine stimulus evoked light modulated neurons
        print('Calculating stimulus evoked light modulated neurons..')
        roc_l_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                             trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                                             trials.loc[trials['signed_contrast'] == -1, 'laser_stimulation'],
                                             post_time=POST_TIME[1])[0]
        roc_r_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                             trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                                             trials.loc[trials['signed_contrast'] == 1, 'laser_stimulation'],
                                             post_time=POST_TIME[1])[0]
        roc_permut_l_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_r_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            pseudo_laser = np.round(generate_pseudo_blocks(trials.shape[0], first5050=0))
            roc_permut_l_light[k, :] = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                pseudo_laser[trials['signed_contrast'] == -1], post_time=POST_TIME[1])[0]
            roc_permut_r_light[k, :] = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                pseudo_laser[trials['signed_contrast'] == 1], post_time=POST_TIME[1])[0]

        # Add results to df
        stim_neurons = stim_neurons.append(pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
            'cluster_id': clusters_pass, 'region': cluster_regions,
            'roc_l_stim': roc_l_stim, 'roc_r_stim': roc_r_stim,
            'mod_l_stim': ((roc_l_stim > np.percentile(roc_permut_l_stim, 97.5, axis=0))
                           | (roc_l_stim < np.percentile(roc_permut_l_stim, 2.5, axis=0))),
            'enh_l_stim': roc_l_stim > np.percentile(roc_permut_l_stim, 97.5, axis=0),
            'supp_l_stim': roc_l_stim < np.percentile(roc_permut_l_stim, 2.5, axis=0),
            'mod_r_stim': ((roc_r_stim > np.percentile(roc_permut_r_stim, 97.5, axis=0))
                           | (roc_r_stim < np.percentile(roc_permut_r_stim, 2.5, axis=0))),
            'enh_r_stim': roc_r_stim > np.percentile(roc_permut_r_stim, 97.5, axis=0),
            'supp_r_stim': roc_r_stim < np.percentile(roc_permut_r_stim, 2.5, axis=0),
            'roc_l_light': roc_l_light, 'roc_r_light': roc_r_light,
            'mod_l_light': ((roc_l_light > np.percentile(roc_permut_l_light, 97.5, axis=0))
                           | (roc_l_light < np.percentile(roc_permut_l_light, 2.5, axis=0))),
            'enh_l_light': roc_l_light > np.percentile(roc_permut_l_light, 97.5, axis=0),
            'supp_l_light': roc_l_light < np.percentile(roc_permut_l_light, 2.5, axis=0),
            'mod_r_light': ((roc_r_light > np.percentile(roc_permut_r_light, 97.5, axis=0))
                           | (roc_r_light < np.percentile(roc_permut_r_light, 2.5, axis=0))),
            'enh_r_light': roc_r_light > np.percentile(roc_permut_r_light, 97.5, axis=0),
            'supp_r_light': roc_r_light < np.percentile(roc_permut_r_light, 2.5, axis=0)}))

stim_neurons.to_csv(join(save_path, 'stim_light_modulated_neurons.csv'))
