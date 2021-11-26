#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()
log_reg = LogisticRegression(penalty='none', random_state=42)

# Settings
RT_CUTOFF = 0.4
REWARD_WIN = 10  # trials
MIN_TRIALS = 5  # for estimating reward bias
K_FOLD = 5
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ModelAgnostic')

results_df = pd.DataFrame()
kf = KFold(n_splits=K_FOLD)
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    all_trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            else:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue

        for t in range(REWARD_WIN+1, trials.shape[0]):
            trials_slice = trials[t-(REWARD_WIN+1):t-1]

            # reward bias opto
            stim_trials = trials_slice[trials_slice['laser_stimulation'] == 1]
            if stim_trials.shape[0] >= MIN_TRIALS:
                rew_win = np.zeros(stim_trials.shape[0])
                rew_win[(stim_trials['choice'] == -1) & (stim_trials['feedbackType'] == 1)] = -1
                rew_win[(stim_trials['choice'] == -1) & (stim_trials['feedbackType'] == -1)] = 1
                rew_win[(stim_trials['choice'] == 1) & (stim_trials['feedbackType'] == 1)] = 1
                rew_win[(stim_trials['choice'] == 1) & (stim_trials['feedbackType'] == -1)] = -1
                trials.loc[t, 'rew_bias_opto'] = np.sum(rew_win)

            # reward bias no opto
            no_stim_trials = trials_slice[trials_slice['laser_stimulation'] == 0]
            if no_stim_trials.shape[0] >= MIN_TRIALS:
                rew_win = np.zeros(no_stim_trials.shape[0])
                rew_win[(no_stim_trials['choice'] == -1) & (no_stim_trials['feedbackType'] == 1)] = -1
                rew_win[(no_stim_trials['choice'] == -1) & (no_stim_trials['feedbackType'] == -1)] = 1
                rew_win[(no_stim_trials['choice'] == 1) & (no_stim_trials['feedbackType'] == 1)] = 1
                rew_win[(no_stim_trials['choice'] == 1) & (no_stim_trials['feedbackType'] == -1)] = -1
                rew_bias_no_opto = np.sum(rew_win)
                trials.loc[t, 'rew_bias_no_opto'] = np.sum(rew_win)

        # Add to trial dataframe
        all_trials = all_trials.append(trials, ignore_index=True)

    # Predict choices with logistic regression
    rew_bias_opto = all_trials.loc[~all_trials['rew_bias_opto'].isnull()
                                   & (all_trials['reaction_times'] > RT_CUTOFF), 'rew_bias_opto'].values.reshape(-1, 1)
    choice = all_trials.loc[~all_trials['rew_bias_opto'].isnull()
                            & (all_trials['reaction_times'] > RT_CUTOFF), 'choice'].values
    pred = np.zeros(choice.shape[0])
    for train_index, test_index in kf.split(rew_bias_opto):
        log_reg.fit(rew_bias_opto[train_index], choice[train_index])
        pred[test_index] = log_reg.predict(rew_bias_opto[test_index])
    opto_long = accuracy_score(choice, pred)

    rew_bias_opto = all_trials.loc[~all_trials['rew_bias_opto'].isnull()
                                   & (all_trials['reaction_times'] < RT_CUTOFF), 'rew_bias_opto'].values.reshape(-1, 1)
    choice = all_trials.loc[~all_trials['rew_bias_opto'].isnull()
                            & (all_trials['reaction_times'] < RT_CUTOFF), 'choice'].values
    pred = np.zeros(choice.shape[0])
    for train_index, test_index in kf.split(rew_bias_opto):
        log_reg.fit(rew_bias_opto[train_index], choice[train_index])
        pred[test_index] = log_reg.predict(rew_bias_opto[test_index])
    opto_short = accuracy_score(choice, pred)

    rew_bias_opto = all_trials.loc[~all_trials['rew_bias_no_opto'].isnull()
                                   & (all_trials['reaction_times'] > RT_CUTOFF), 'rew_bias_no_opto'].values.reshape(-1, 1)
    choice = all_trials.loc[~all_trials['rew_bias_no_opto'].isnull()
                            & (all_trials['reaction_times'] > RT_CUTOFF), 'choice'].values
    pred = np.zeros(choice.shape[0])
    for train_index, test_index in kf.split(rew_bias_opto):
        log_reg.fit(rew_bias_opto[train_index], choice[train_index])
        pred[test_index] = log_reg.predict(rew_bias_opto[test_index])
    no_opto_long = accuracy_score(choice, pred)

    rew_bias_opto = all_trials.loc[~all_trials['rew_bias_no_opto'].isnull()
                                   & (all_trials['reaction_times'] < RT_CUTOFF), 'rew_bias_no_opto'].values.reshape(-1, 1)
    choice = all_trials.loc[~all_trials['rew_bias_no_opto'].isnull()
                            & (all_trials['reaction_times'] < RT_CUTOFF), 'choice'].values
    pred = np.zeros(choice.shape[0])
    for train_index, test_index in kf.split(rew_bias_opto):
        log_reg.fit(rew_bias_opto[train_index], choice[train_index])
        pred[test_index] = log_reg.predict(rew_bias_opto[test_index])
    no_opto_short = accuracy_score(choice, pred)

    results_df = results_df.append(pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'long_rt': [opto_long, no_opto_long], 'short_rt': [opto_short, no_opto_short],
        'opto': [1, 0]}), ignore_index=True)

# %% Plot

plot_df = results_df.copy()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] = 'Sert'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] = 'WT'
plot_df.loc[plot_df['opto'] == 1, 'opto'] = 'Stim'
plot_df.loc[plot_df['opto'] == 0, 'opto'] = 'No stim'

colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.lineplot(x='opto', y='long_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax1)
ax1.legend(frameon=False)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Reward history prediction of choice',
        title=f'Long RT (> {RT_CUTOFF}s)', ylim=[0, 1])

sns.lineplot(x='opto', y='short_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend=None, dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Reward history prediction of choice',
        title=f'Short RT (< {RT_CUTOFF}s)', ylim=[0, 1])

sns.barplot(x='sert-cre', y='long_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax3)
ax3.set(xlabel='', ylabel='Reward history prediction of choice', ylim=[0, 1])
ax3.legend(frameon=False, bbox_to_anchor=(0.65, 0.8))
#ax3.get_legend().remove()

sns.barplot(x='sert-cre', y='short_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax4)
ax4.set(xlabel='', ylabel='Reward history prediction of choice', ylim=[0, 1])
ax4.get_legend().remove()

plt.tight_layout(pad=3)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'log_regression_reward_history'), dpi=300)

