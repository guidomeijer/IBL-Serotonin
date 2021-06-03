#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:36:23 2020

@author: guido
"""

import pandas as pd
import seaborn as sns
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from functions_pharmacology import paths, plot_psychometric
from ibl_pipeline import subject, acquisition, behavior

# Load in session dates
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)
mice = sessions['Nickname'].unique()

# Initialize plot
f, ax = plt.subplots(1, sessions['Nickname'].unique().shape[0] + 1, figsize=(20, 5))
sns.set(style="ticks", context="paper", font_scale=1.5)
colors = sns.color_palette(n_colors=3)

for j, condition in enumerate(['Pre-vehicle', 'Drug']):
    for i, nickname in enumerate(sessions['Nickname'].unique()):
        trials = pd.DataFrame()
        for k, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, condition].values):

            # Load in data
            ses_query = (acquisition.Session * subject.Subject
                         & 'subject_nickname = "%s"' % nickname
                         & 'date(session_start_time) = "%s"' % date
                         & 'task_protocol LIKE "%biased%"')
            assert len(ses_query) == 1
            this_trials = (ses_query * behavior.TrialSet.Trial).fetch(format='frame').reset_index()
            trials = trials.append(this_trials)

        # Restructure into input for psychometric function plotting
        trials['signed_contrast'] = (trials['trial_stim_contrast_right']
                                     - trials['trial_stim_contrast_left']) * 100
        trials.loc[trials['trial_response_choice'] == 'CW', 'right_choice'] = 0
        trials.loc[trials['trial_response_choice'] == 'CCW', 'right_choice'] = 1

        # Get data for 50/50 trials
        stim_levels = trials[trials['trial_stim_prob_left'] == 0.5].groupby(
                                                'signed_contrast').size().index.values
        n_trials = trials[trials['trial_stim_prob_left'] == 0.5].groupby(
                                                'signed_contrast').size().values
        prop_right = trials[trials['trial_stim_prob_left'] == 0.5].groupby(
                                                'signed_contrast').mean()['right_choice'].values

        # Plot psychometric curve
        plot_psychometric(stim_levels, n_trials, prop_right, ax=ax[i], color=colors[j])
        ax[i].set(xlabel='Signed contrast (%)', ylabel='Rightward responses', title=nickname)
        ax[i].legend(['Vehicle', '_', '_', 'Drug'], frameon=False)

        # Add to arrays
        if i == 0:
            all_trials = n_trials
            all_prop = prop_right
        else:
            all_trials = np.vstack((all_trials, n_trials))
            all_prop = np.vstack((all_prop, prop_right))

    # Plot over animals curve
    plot_psychometric(stim_levels, all_trials, all_prop, ax=ax[-1], color=colors[j])
    ax[-1].set(xlabel='Signed contrast (%)', ylabel='Rightward responses', title='All mice')
    ax[-1].legend(['Vehicle', '_', '_', 'Drug'], frameon=False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(paths()[1], 'altanserin_psychometric_curves'))
