#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:09:11 2023
By: Guido Meijer
"""
import numpy as np
import pandas as pd
from os.path import join
from scipy.stats import ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (load_subjects, query_opto_sessions, behavioral_criterion,
                                 load_trials, figure_style, paths)
from one.api import ONE
one = ONE()

# Settings
MIN_TRIALS = 800
SINGLE_TRIALS = [2, 3]

# Query which subjects to use and create eid list per subject
subjects = load_subjects()

# Get paths
fig_path = join(paths()[0], 'Behavior')

p_repeat_df, p_repeat_bins_df = pd.DataFrame(), pd.DataFrame()
p_repeat_probe_df, p_repeat_single_df = pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(subjects['subject']):

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    #eids = behavioral_criterion(eids, min_perf=0.7, min_trials=200, verbose=False, one=one)

    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
        except Exception as err:
            print(err)
            continue
        if np.sum(these_trials['laser_probability'] == 0.5) > 0:
            continue
        these_trials['trial'] = these_trials.index.values
        these_trials['session'] = j
        trials = pd.concat((trials, these_trials), ignore_index=True)
    if trials.shape[0] < MIN_TRIALS:
        continue
    print(f'{subject}: {trials.shape[0]} trials')

    # Get repeated choices of rewarded and not rewarded trials
    rep_rew = (trials['choice'].values[:-1] == trials['choice'].values[1:]) & (trials['correct'].values[:-1] == 1)
    trials['repeat_rew'] = np.concatenate(([False], rep_rew)).astype(int)
    rep_no_rew = (trials['choice'].values[:-1] == trials['choice'].values[1:]) & (trials['correct'].values[:-1] == 0)
    trials['repeat_no_rew'] = np.concatenate(([False], rep_no_rew)).astype(int)

    # Get P(repeat choice) centered at probe trials
    this_probe_df = pd.DataFrame()
    for s in np.unique(trials['session']):
        this_ses = trials[trials['session'] == s].reset_index(drop=True)
        opto_probe_ind = this_ses[this_ses['signed_contrast'] == 0].index
        for b, trial_ind in enumerate(opto_probe_ind):
            
            # Single trials
            these_rews = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                      'repeat_rew'].values
            
            these_no_rews = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                         'repeat_no_rew'].values
            this_ses['rel_trial'] = this_ses['trial'] - trial_ind
            these_trials = this_ses.loc[trial_ind-SINGLE_TRIALS[0]:trial_ind+SINGLE_TRIALS[-1],
                                        'rel_trial']
            this_probe_df = pd.concat((this_probe_df, pd.DataFrame(data={
                'repeat_rew': these_rews, 'repeat_no_rew': these_no_rews, 'trial': these_trials})))            
   
    # Add to dataframe
    this_repeats = this_probe_df.groupby('trial').mean()['repeat_rew'].values * 100
    p_repeat_probe_df = pd.concat((p_repeat_probe_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'rewarded': 1})))
    this_repeats = this_probe_df.groupby('trial').mean()['repeat_no_rew'].values * 100
    p_repeat_probe_df = pd.concat((p_repeat_probe_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:SINGLE_TRIALS[0]]),
        'trial': np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject,
        'rewarded': 0})))
     

# %% Plot

colors, dpi = figure_style()
plot_colors = [colors['wt'], colors['sert']]
plot_labels = ['WT', 'SERT']
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

#ax1.plot([-SINGLE_TRIALS[0], SINGLE_TRIALS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_repeat_probe_df, x='trial', y='p_repeat',
             err_style='bars', errorbar='se', hue='rewarded', hue_order=[0, 1],
             palette=[colors['no-stim'], colors['sert']], ax=ax1)
#ax4.scatter(np.unique(p_repeat_probe_df['trial'])[p_probe < 0.05],
#            np.ones(np.sum(p_probe < 0.05))*3,
#            marker='*', color='k')
handles, labels = ax1.get_legend_handles_labels()
labels = ['No reward', 'Reward']
ax1.legend(handles, labels, frameon=False, prop={'size': 5}, loc='lower left')
#ax1.set(ylabel='P[repeat choice] (%)', xlabel='Trials since 0% contrast trial',
#        xticks=np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[-1]+1), yticks=[-10, 0, 10])
ax1.set(ylabel='P[repeat choice] (%)', xlabel='Trials since 0% contrast trial',
        xticks=np.arange(-SINGLE_TRIALS[0], SINGLE_TRIALS[-1]+1), ylim=[0, 70])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'prev_rew_choice_0_contrast.jpg'), dpi=600)


