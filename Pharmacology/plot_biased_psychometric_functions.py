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
from scipy.stats import sem
import matplotlib.pyplot as plt
from functions_pharmacology import paths, plot_psychometric
from ibl_pipeline import subject, acquisition, behavior

# Load in session dates
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)
mice = sessions['Nickname'].unique()

# Initialize plot
f1, ax = plt.subplots(1, sessions['Nickname'].unique().shape[0] + 1, figsize=(20, 5))
sns.set(style="ticks", context="paper", font_scale=1.5)
f2, ax1 = plt.subplots(1, 1, figsize=(6, 5))
sns.set(style="ticks", context="paper", font_scale=1.5)
colors = sns.color_palette('Dark2', n_colors=2)

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

        # Get data for 20/80 trials
        stim_levels_20 = trials[trials['trial_stim_prob_left'] == 0.2].groupby(
                                                'signed_contrast').size().index.values
        n_trials_20 = trials[trials['trial_stim_prob_left'] == 0.2].groupby(
                                                'signed_contrast').size().values
        prop_right_20 = trials[trials['trial_stim_prob_left'] == 0.2].groupby(
                                                'signed_contrast').mean()['right_choice'].values

        # Get data for 80/20 trials
        stim_levels_80 = trials[trials['trial_stim_prob_left'] == 0.8].groupby(
                                                'signed_contrast').size().index.values
        n_trials_80 = trials[trials['trial_stim_prob_left'] == 0.8].groupby(
                                                'signed_contrast').size().values
        prop_right_80 = trials[trials['trial_stim_prob_left'] == 0.8].groupby(
                                                'signed_contrast').mean()['right_choice'].values

        # Plot psychometric curve
        plot_psychometric(stim_levels_20, n_trials_20, prop_right_20, ax=ax[i], color=colors[j])
        plot_psychometric(stim_levels_80, n_trials_80, prop_right_80, ax=ax[i], color=colors[j])
        ax[i].set(xlabel='Signed contrast (%)', ylabel='Rightward choices', title=nickname)
        # ax[i].legend(['Vehicle', '_', '_', 'Drug'], frameon=False)

        # Add to arrays
        if i == 0:
            all_trials_20 = n_trials_20
            all_prop_20 = prop_right_20
            all_trials_80 = n_trials_80
            all_prop_80 = prop_right_80
        else:
            all_trials_20 = np.vstack((all_trials_20, n_trials_20))
            all_prop_20 = np.vstack((all_prop_20, prop_right_20))
            all_trials_80 = np.vstack((all_trials_80, n_trials_80))
            all_prop_80 = np.vstack((all_prop_80, prop_right_80))

    # Plot over animals curve
    plot_psychometric(stim_levels_20, all_trials_20, all_prop_20, ax=ax[-1], color=colors[j])
    plot_psychometric(stim_levels_80, all_trials_80, all_prop_80, ax=ax[-1], color=colors[j])
    ax[-1].set(xlabel='Signed contrast (%)', ylabel='Rightward responses', title='All mice')
    # ax[-1].legend(['Vehicle', '_', '_', 'Drug'], frameon=False)

    # Plot difference
    ax1.errorbar(stim_levels_20, np.mean(all_prop_20 - all_prop_80, axis=0),
                 yerr=sem(all_prop_20 - all_prop_80), color=colors[j], lw=2, label=condition)
    ax1.scatter(stim_levels_20, np.mean(all_prop_20 - all_prop_80, axis=0), color=colors[j], s=60)

ax1.set(ylabel='\u0394 Rightward choices', xlabel='Signed contrast (%)',
        xticks=[-35, -25, -12.5, 0, 12.5, 25, 35],
        xticklabels=['-100', '-25', '-12.5', '0', '12.5', '25', '100'])
ax1.legend()
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(paths()[1], 'altanserin_bias_curves'), dpi=300)

plt.figure(f1.number)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(paths()[1], 'altanserin_psychometric_curves'), dpi=300)
