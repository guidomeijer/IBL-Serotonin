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
TRIALS_AFTER_SWITCH = 6
REWARD_WIN = 10  # trials
CHOICE_WIN = 5  # trials
MIN_TRIALS = 5  # for estimating reward bias
subjects = load_subjects(behavior=True)
fig_path, save_path = paths()
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
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True, one=one)
            else:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except:
            print(f'\nError loading {eid}\n')
            continue
        
        if trials.shape[0] < 100:
            continue

        # Make array of after block switch trials
        trials['block_switch'] = np.zeros(trials.shape[0])
        trial_blocks = (trials['probabilityLeft'] == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
        block_trans = np.append(block_trans, [trial_blocks.shape[0]])
        for s, ind in enumerate(block_trans):
            trials.loc[ind:ind+TRIALS_AFTER_SWITCH, 'block_switch'] = 1

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

            # choice bias early block
            trials_slice = trials[t+REWARD_WIN+1:t+REWARD_WIN+CHOICE_WIN+1]
            early_block_trials = trials_slice[trials_slice['block_switch'] == 1]
            if early_block_trials.shape[0] > 0:
                trials.loc[t, 'choice_bias_early'] = np.sum(early_block_trials['choice'])

            # choice bias long rt
            late_block_trials = trials_slice[trials_slice['block_switch'] == 0]
            if late_block_trials.shape[0] > 0:
                trials.loc[t, 'choice_bias_late'] = np.sum(late_block_trials['choice'])

        # Correlate choice and reward bias
        r_opto_early = pearsonr(trials.dropna(subset=['rew_bias_opto', 'choice_bias_early'])['rew_bias_opto'],
                                trials.dropna(subset=['rew_bias_opto', 'choice_bias_early'])['choice_bias_early'])[0]
        r_no_opto_early = pearsonr(trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_early'])['rew_bias_no_opto'],
                                   trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_early'])['choice_bias_early'])[0]
        r_opto_late = pearsonr(trials.dropna(subset=['rew_bias_opto', 'choice_bias_late'])['rew_bias_opto'],
                               trials.dropna(subset=['rew_bias_opto', 'choice_bias_late'])['choice_bias_late'])[0]
        r_no_opto_late = pearsonr(trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_late'])['rew_bias_no_opto'],
                                  trials.dropna(subset=['rew_bias_no_opto', 'choice_bias_late'])['choice_bias_late'])[0]

        results_df = pd.concat((results_df, pd.DataFrame(data={
            'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
            'date': one.get_details(eid)['date'], 'eid': eid,
            'r_early_block': [r_opto_early, r_no_opto_early], 'r_late_block': [r_opto_late, r_no_opto_late],
            'opto': [1, 0]})), ignore_index=True)

# %% Plot

plot_df = results_df.groupby(['subject', 'opto']).median().reset_index()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] = 'Sert'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] = 'WT'
plot_df.loc[plot_df['opto'] == 1, 'opto'] = 'Stim'
plot_df.loc[plot_df['opto'] == 0, 'opto'] = 'No stim'

colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.lineplot(x='opto', y='r_early_block', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax1)
ax1.legend(frameon=False)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No opto', 'Opto'], ylabel='Corr. reward bias vs choice bias',
        title=f'Early block (<= {TRIALS_AFTER_SWITCH} trials)', ylim=[-0.61, 0.2])

sns.lineplot(x='opto', y='r_late_block', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend=None, dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No opto', 'Opto'], ylabel='Corr. reward bias vs choice bias',
        title=f'Late block (> {TRIALS_AFTER_SWITCH} trials)', ylim=[0, 0.8])

sns.barplot(x='sert-cre', y='r_early_block', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax3)
ax3.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[-0.61, 0.2])
ax3.legend(frameon=False)

sns.barplot(x='sert-cre', y='r_late_block', data=plot_df, hue='opto', palette=[colors['no-stim'], colors['stim']],
            order=['WT', 'Sert'], ax=ax4)
ax4.set(xlabel='', ylabel='Corr. reward bias vs choice bias', ylim=[0, 0.8])
ax4.get_legend().remove()

plt.tight_layout(pad=3)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'early_block_late_block'), dpi=300)

