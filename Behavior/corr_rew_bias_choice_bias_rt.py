#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:53:33 2021
By: Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()

# Settings
RT_CUTOFF = 0.5
REWARD_WIN = 10  # trials
CHOICE_WIN = 5  # trials
MIN_TRIALS = 5  # for estimating reward bias
subjects = load_subjects(behavior=True)
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ModelAgnostic')

results_df = pd.DataFrame()
all_trials = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    for j, eid in enumerate(eids):
        try:
            if subjects.loc[i, 'sert-cre'] == 1:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            else:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            continue

        for t in range(trials.shape[0] - (REWARD_WIN + CHOICE_WIN)):
            trials_slice = trials[t:t+REWARD_WIN]

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

            # choice bias short rt
            trials_slice = trials[t+REWARD_WIN+1:t+REWARD_WIN+CHOICE_WIN+1]
            short_rt_trials = trials_slice[trials_slice['reaction_times'] < RT_CUTOFF]
            trials.loc[t, 'choice_bias_short'] = np.sum(short_rt_trials['choice'])

            # choice bias long rt
            long_rt_trials = trials_slice[trials_slice['reaction_times'] > RT_CUTOFF]
            if long_rt_trials.shape[0] > 0:
                trials.loc[t, 'choice_bias_long'] = np.sum(long_rt_trials['choice'])

        # Correlate choice and reward bias
        r_opto_long = pearsonr(trials.dropna(subset=['rew_bias_opto', 'choice_bias_long'])['rew_bias_opto'],
                               trials.dropna(subset=['rew_bias_opto', 'choice_bias_long'])['choice_bias_long'])[0]
        r_no_opto_long = pearsonr(trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_long'])['rew_bias_no_opto'],
                                  trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_long'])['choice_bias_long'])[0]
        r_opto_short = pearsonr(trials.dropna(subset=['rew_bias_opto', 'choice_bias_short'])['rew_bias_opto'],
                                trials.dropna(subset=['rew_bias_opto', 'choice_bias_short'])['choice_bias_short'])[0]
        r_no_opto_short = pearsonr(trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_short'])['rew_bias_no_opto'],
                                   trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_short'])['choice_bias_short'])[0]

        results_df = results_df.append(pd.DataFrame(data={
            'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
            'date': one.get_details(eid)['date'], 'eid': eid,
            'r_long_rt': [r_opto_long, r_no_opto_long], 'r_short_rt': [r_opto_short, r_no_opto_short],
            'opto': [1, 0]}), ignore_index=True)

# %% Plot

plot_df = results_df.groupby(['subject', 'opto']).median().reset_index()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] = 'Sert'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] = 'WT'
plot_df.loc[plot_df['opto'] == 1, 'opto'] = 'Stim'
plot_df.loc[plot_df['opto'] == 0, 'opto'] = 'No stim'

colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.lineplot(x='opto', y='r_long_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax1)
ax1.legend(frameon=False)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Corr. reward bias vs choice bias',
        title=f'Long RT (> {RT_CUTOFF}s)', ylim=[0, 0.5])

sns.lineplot(x='opto', y='r_short_rt', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend=None, dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Corr. reward bias vs choice bias',
        title=f'Short RT (< {RT_CUTOFF}s)', ylim=[0, 0.8])

sns.barplot(x='sert-cre', y='r_long_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax3)
ax3.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[0, 0.5])
ax3.legend(frameon=False, bbox_to_anchor=(0.65, 0.8))
#ax3.get_legend().remove()

sns.barplot(x='sert-cre', y='r_short_rt', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax4)
ax4.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[0, 0.8])
ax4.get_legend().remove()

plt.tight_layout(pad=3)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'rew_bias_choice_bias'), dpi=300)

