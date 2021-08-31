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
                                       generate_pseudo_stimuli, generate_pseudo_blocks,
                                       generate_pseudo_session)
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import paths, remap, query_ephys_sessions, load_trials, figure_style
from one.api import ONE
one = ONE()

# Settings
OVERWRITE = True
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME = [0, 0.5]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'light-modulated-neurons')
save_path = join(save_path)

# Query sessions
eids, _ = query_ephys_sessions(selection='all', one=one)

if OVERWRITE:
    stim_neurons = pd.DataFrame(columns=['subject', 'date'])
else:
    stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons_no_histology.csv'))
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    if (subject in stim_neurons['subject'].values) and (date in stim_neurons['date'].values) and (OVERWRITE is False):
        continue

    # Load trials dataframe
    try:
        trials = load_trials(eid, laser_stimulation=True, one=one)
    except:
        print('cannot load trials')
        continue
    if trials is None:
        continue
    if trials.shape[0] < 200:
        continue

    # Load in spikes
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)
    except:
        continue

    for p, probe in enumerate(spikes.keys()):
        if spikes[probe] is None:
            continue

        # Filter neurons that pass QC
        if 'metrics' in clusters[probe].keys():
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]

        # Determine stimulus responsive neurons
        print('Calculating stimulus responsive neurons..')
        roc_l_stim = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                      trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME)[0]
        roc_l_stim = 2 * (roc_l_stim - 0.5)
        roc_r_stim = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                      trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME)[0]
        roc_r_stim = 2 * (roc_r_stim - 0.5)

        # Determine 0% contrast responsive neurons
        print('Calculating stimulus responsive neurons..')
        roc_0_stim = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                      trials.loc[trials['signed_contrast'] == 0, 'stimOn_times'],
                                      pre_time=PRE_TIME, post_time=POST_TIME)[0]
        roc_0_stim = 2 * (roc_0_stim - 0.5)

        # Do permutation testing for stimulus
        roc_permut_l_stim = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_r_stim = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_0_stim = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            _, contrast_l, contrast_r = generate_pseudo_stimuli(trials.shape[0], first5050=0)
            roc_permut_l_stim[k, :] = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                       trials.loc[contrast_l == 1, 'stimOn_times'],
                                                       pre_time=PRE_TIME, post_time=POST_TIME)[0]
            roc_permut_r_stim[k, :] = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                       trials.loc[contrast_r == 1, 'stimOn_times'],
                                                       pre_time=PRE_TIME, post_time=POST_TIME)[0]
            roc_permut_0_stim[k, :] = roc_single_event(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[((contrast_l == 0) | (contrast_r == 0)), 'stimOn_times'],
                pre_time=PRE_TIME, post_time=POST_TIME)[0]

        # Determine evoked light modulated neurons
        print('Calculating task event evoked light modulated neurons..')
        roc_l_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                             trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                                             trials.loc[trials['signed_contrast'] == -1, 'laser_stimulation'],
                                             post_time=POST_TIME[1])[0]
        roc_l_light = 2 * (roc_l_light - 0.5)
        roc_r_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                             trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                                             trials.loc[trials['signed_contrast'] == 1, 'laser_stimulation'],
                                             post_time=POST_TIME[1])[0]
        roc_r_light = 2 * (roc_r_light - 0.5)
        roc_0_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                             trials.loc[trials['signed_contrast'] == 0, 'stimOn_times'],
                                             trials.loc[trials['signed_contrast'] == 0, 'laser_stimulation'],
                                             post_time=POST_TIME[1])[0]
        roc_0_light = 2 * (roc_0_light - 0.5)
        """
        roc_rew_light = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                               trials.loc[trials['correct'] == 1, 'feedback_times'],
                                               trials.loc[trials['correct'] == 1, 'laser_stimulation'],
                                               post_time=POST_TIME[1])[0]
        roc_rew_light = 2 * (roc_rew_light - 0.5)
        """

        # Do permutation testing
        roc_permut_l_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_r_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_0_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        roc_permut_rew_light = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            pseudo_laser = np.round(generate_pseudo_blocks(trials.shape[0], first5050=0))
            this_roc_permut_l_light = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['signed_contrast'] == -1, 'stimOn_times'],
                pseudo_laser[trials['signed_contrast'] == -1], post_time=POST_TIME[1])[0]
            roc_permut_l_light[k, :] = 2 * (this_roc_permut_l_light - 0.5)
            this_roc_permut_r_light = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['signed_contrast'] == 1, 'stimOn_times'],
                pseudo_laser[trials['signed_contrast'] == 1], post_time=POST_TIME[1])[0]
            roc_permut_r_light[k, :] = 2 * (this_roc_permut_r_light - 0.5)
            this_roc_permut_0_light = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['signed_contrast'] == 0, 'stimOn_times'],
                pseudo_laser[trials['signed_contrast'] == 0], post_time=POST_TIME[1])[0]
            roc_permut_0_light[k, :] = 2 * (this_roc_permut_0_light - 0.5)
            """
            this_roc_permut_rew_light = roc_between_two_events(
                spikes[probe].times, spikes[probe].clusters,
                trials.loc[trials['correct'] == 1, 'feedback_times'],
                pseudo_laser[trials['correct'] == 1], post_time=POST_TIME[1])[0]
            roc_permut_rew_light[k, :] = 2 * (this_roc_permut_rew_light - 0.5)


        # Determine reward responsive neurons
        print('Calculating reward responsive neurons..')
        roc_reward = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                            trials.loc[trials['correct'] == 0, 'feedback_times'],
                                            trials.loc[trials['correct'] == 1, 'feedback_times'],
                                            post_time=POST_TIME[1])[0]
        roc_reward = 2 * (roc_reward - 0.5)
        roc_permut_reward = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            pseudo_trials = generate_pseudo_session(trials)
            roc_permut_reward = roc_between_two_events(spikes[probe].times, spikes[probe].clusters,
                                                     trials.loc[pseudo_trials['correct'] == 0, 'feedback_times'],
                                                     trials.loc[pseudo_trials['correct'] == 1, 'feedback_times'],
                                                     post_time=POST_TIME[1])[0]
            roc_permut_reward[k, :] = 2 * (roc_permut_reward - 0.5)
        """

        # Add results to df
        stim_neurons = stim_neurons.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
                'cluster_id': clusters_pass,
                'roc_l_stim': roc_l_stim, 'roc_r_stim': roc_r_stim, 'roc_0_stim': roc_0_stim,
                'mod_l_stim': ((roc_l_stim > np.percentile(roc_permut_l_stim, 97.5, axis=0))
                               | (roc_l_stim < np.percentile(roc_permut_l_stim, 2.5, axis=0))),
                'enh_l_stim': roc_l_stim > np.percentile(roc_permut_l_stim, 97.5, axis=0),
                'supp_l_stim': roc_l_stim < np.percentile(roc_permut_l_stim, 2.5, axis=0),
                'mod_r_stim': ((roc_r_stim > np.percentile(roc_permut_r_stim, 97.5, axis=0))
                               | (roc_r_stim < np.percentile(roc_permut_r_stim, 2.5, axis=0))),
                'enh_r_stim': roc_r_stim > np.percentile(roc_permut_r_stim, 97.5, axis=0),
                'supp_r_stim': roc_r_stim < np.percentile(roc_permut_r_stim, 2.5, axis=0),
                'mod_0_stim': ((roc_0_stim > np.percentile(roc_permut_0_stim, 97.5, axis=0))
                               | (roc_0_stim < np.percentile(roc_permut_0_stim, 2.5, axis=0))),
                'enh_0_stim': roc_0_stim > np.percentile(roc_permut_0_stim, 97.5, axis=0),
                'supp_0_stim': roc_0_stim < np.percentile(roc_permut_0_stim, 2.5, axis=0),
                'roc_l_light': roc_l_light, 'roc_r_light': roc_r_light, 'roc_0_light': roc_0_light,
                'mod_l_light': ((roc_l_light > np.percentile(roc_permut_l_light, 97.5, axis=0))
                               | (roc_l_light < np.percentile(roc_permut_l_light, 2.5, axis=0))),
                'enh_l_light': roc_l_light > np.percentile(roc_permut_l_light, 97.5, axis=0),
                'supp_l_light': roc_l_light < np.percentile(roc_permut_l_light, 2.5, axis=0),
                'mod_r_light': ((roc_r_light > np.percentile(roc_permut_r_light, 97.5, axis=0))
                               | (roc_r_light < np.percentile(roc_permut_r_light, 2.5, axis=0))),
                'enh_r_light': roc_r_light > np.percentile(roc_permut_r_light, 97.5, axis=0),
                'supp_r_light': roc_r_light < np.percentile(roc_permut_r_light, 2.5, axis=0),
                'mod_0_light': ((roc_0_light > np.percentile(roc_permut_0_light, 97.5, axis=0))
                               | (roc_0_light < np.percentile(roc_permut_0_light, 2.5, axis=0))),
                'enh_0_light': roc_0_light > np.percentile(roc_permut_0_light, 97.5, axis=0),
                'supp_0_light': roc_0_light < np.percentile(roc_permut_0_light, 2.5, axis=0)}))

    """
    'mod_reward': ((roc_reward > np.percentile(roc_permut_reward, 97.5, axis=0))
                   | (roc_reward < np.percentile(roc_permut_reward, 2.5, axis=0))),
    'enh_reward': roc_reward > np.percentile(roc_permut_reward, 97.5, axis=0),
    'supp_reward': roc_reward < np.percentile(roc_permut_reward, 2.5, axis=0),
    'mod_rew_light': ((roc_rew_light > np.percentile(roc_permut_rew_light, 97.5, axis=0))
                   | (roc_rew_light < np.percentile(roc_permut_rew_light, 2.5, axis=0))),
    'enh_rew_light': roc_rew_light > np.percentile(roc_permut_rew_light, 97.5, axis=0),
    'supp_rew_light': roc_rew_light < np.percentile(roc_permut_rew_light, 2.5, axis=0)
    """
    stim_neurons.to_csv(join(save_path, 'stim_light_modulated_neurons_no_histology.csv'))
stim_neurons.to_csv(join(save_path, 'stim_light_modulated_neurons_no_histology.csv'))
