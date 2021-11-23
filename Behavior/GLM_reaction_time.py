#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()

# Settings
RT_CUTOFF = 0.5
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

results_single_df = pd.DataFrame()
results_df = pd.DataFrame()
all_trials = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True,
                                           invert_choice=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False,
                                           invert_choice=True, one=one)
        except:
            continue
        these_trials = these_trials.rename(columns={'feedbackType': 'trial_feedback_type'})
        these_trials['block_id'] = (these_trials['probabilityLeft'] == 0.2).astype(int)
        these_trials['stimulus_side'] = these_trials['stim_side'].copy()
        these_trials.loc[these_trials['signed_contrast'] == 0, 'stimulus_side'] = 0
        these_trials['contrast'] = these_trials['signed_contrast'].abs() * 100
        these_trials['previous_outcome'] = these_trials['trial_feedback_type'].shift(periods=1)
        these_trials['previous_choice'] = these_trials['choice'].shift(periods=1)
        trials = trials.append(these_trials, ignore_index=True)
        all_trials = trials.append(these_trials, ignore_index=True)

    # Remove no-go trials
    trials = trials[trials['choice'] != 0]

    # Fit GLM
    params_all = fit_glm(trials, rt_cutoff=RT_CUTOFF)
    params_short = fit_glm(trials[trials['reaction_times'] > RT_CUTOFF])
    params_long = fit_glm(trials[trials['reaction_times'] < RT_CUTOFF])

    # Add to dataframe
    results_single_df = results_single_df.append(params_all, ignore_index=True)
    results_single_df.loc[results_df.shape[0]-1, 'subject'] = nickname
    results_single_df.loc[results_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_df = results_df.append(params_short, ignore_index=True)
    results_df.loc[results_df.shape[0]-1, 'subject'] = nickname
    results_df.loc[results_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_df.loc[results_df.shape[0]-1, 'reaction_time'] = 'short'
    results_df = results_df.append(params_long, ignore_index=True)
    results_df.loc[results_df.shape[0]-1, 'subject'] = nickname
    results_df.loc[results_df.shape[0]-1, 'sert-cre'] = subjects.loc[i, 'sert-cre']
    results_df.loc[results_df.shape[0]-1, 'reaction_time'] = 'long'

# %% Plot

colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3), dpi=dpi)
xticks = [0.01, 0.05, 0.5, 5, 10, 60]
ax1.hist(np.log10(all_trials['reaction_times']), bins=20)
ax1.set(xlabel='log-transformed reaction times (s)', ylabel='Trials',
        xticks=np.log10(xticks), xticklabels=xticks, xlim=[np.log10(0.01), np.log10(60)])

for i, subject in enumerate(results_df['subject']):
    ax2.plot([1, 2], [results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'short'), 'accuracy'],
                      results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'long'), 'accuracy']],
             color = colors[int(results_df.loc[results_df['subject'] == subject, 'sert-cre'].values[0])], marker='o', ms=2)
ax2.set(xlabel='', xticks=[1, 2], xticklabels=['Short RT', 'Long RT'], ylabel='Model accuracy',
        ylim=[0.6, 0.9])

for i, subject in enumerate(results_df['subject']):
    ax3.plot([1, 2], [results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'short'), '100'],
                      results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'long'), '100']],
             color = colors[int(results_df.loc[results_df['subject'] == subject, 'sert-cre'].values[0])], marker='o', ms=2)
ax3.set(xlabel='', xticks=[1, 2], xticklabels=['Short RT', 'Long RT'], ylabel='100% stimulus weight',
        ylim=[0, 4])

for i, subject in enumerate(results_df['subject']):
    ax4.plot([1, 2], [results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'short'), 'rewarded'],
                      results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'long'), 'rewarded']],
             color = colors[int(results_df.loc[results_df['subject'] == subject, 'sert-cre'].values[0])], marker='o', ms=2)
ax4.set(xlabel='', xticks=[1, 2], xticklabels=['Short RT', 'Long RT'], ylabel='Past choice correct weight',
        ylim=[-0.5, 1.5])

for i, subject in enumerate(results_df['subject']):
    ax5.plot([1, 2], [results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'short'), 'unrewarded'],
                      results_df.loc[(results_df['subject'] == subject) & (results_df['reaction_time'] == 'long'), 'unrewarded']],
             color = colors[int(results_df.loc[results_df['subject'] == subject, 'sert-cre'].values[0])], marker='o', ms=2)
ax5.set(xlabel='', xticks=[1, 2], xticklabels=['Short RT', 'Long RT'], ylabel='Past choice error weight',
        ylim=[-0.5, 1.5])

plt.tight_layout()
sns.despine(trim=True)


