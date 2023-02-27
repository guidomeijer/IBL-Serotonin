#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:09:11 2023
By: Guido Meijer
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (load_subjects, query_opto_sessions, behavioral_criterion,
                                 load_trials, figure_style)
from one.api import ONE
one = ONE()

# Settings
MIN_TRIALS = 800
PROBE_TRIALS = [2, 5]
TRIAL_BINS = np.arange(-10, 31, 5)
trial_bin_size = np.unique(np.diff(TRIAL_BINS))[0]
trial_bin_labels = TRIAL_BINS[:-1] + (np.diff(TRIAL_BINS) / 2)

# Query which subjects to use and create eid list per subject
subjects = load_subjects()

p_repeat_df, p_repeat_bins_df, p_probe_bins_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
p_repeat_probe_df = pd.DataFrame()
for i, subject in enumerate(subjects['subject']):

    # Query sessions
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    eids = query_opto_sessions(subject, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, min_perf=0.8, min_trials=100, verbose=False, one=one)

    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
        except Exception as err:
            print(err)
            continue
        trials = pd.concat((trials, these_trials), ignore_index=True)
    if trials.shape[0] < MIN_TRIALS:
        continue
    print(f'{subject}: {trials.shape[0]} trials')

    # Get repeated choices
    trials['repeat_choice'] = np.concatenate(([False], trials['choice'].values[:-1] == trials['choice'].values[1:])).astype(int)

    # Get probe trial triggered state switches
    trials['probe_trial'] = (trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1)
    opto_probe_ind = np.where(trials['probe_trial'])[0]
    this_probe_df, this_probe_repeat_df = pd.DataFrame(), pd.DataFrame()
    for b, trial_ind in enumerate(opto_probe_ind):
        if (trial_ind + TRIAL_BINS[-1] < trials.shape[0]) & (trial_ind + TRIAL_BINS[0] > 0):

            # Get P(repeat) for the probe trial itself and single trials around it
            these_repeats = trials.loc[trial_ind-PROBE_TRIALS[0]:trial_ind+PROBE_TRIALS[1], 'repeat_choice'].values
            this_probe_repeat_df = pd.concat((this_probe_repeat_df, pd.DataFrame(data={
                'repeat_choice': these_repeats, 'trial': np.arange(-PROBE_TRIALS[0], PROBE_TRIALS[1]+1)})))

            # Get repeats in bins of trials around the probe trial
            these_repeats = np.empty(len(TRIAL_BINS)-1)
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                these_repeats[tt] = np.sum(trials.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]-1,
                                                      'repeat_choice'].values)
            this_probe_df = pd.concat((this_probe_df, pd.DataFrame(data={
                'repeat_choice': these_repeats, 'trial_bin': trial_bin_labels, 'probe_trial': b,
                'trial_ind': np.arange(len(trial_bin_labels))})))
    this_probe_df['repeat_choice'] /= trial_bin_size

    # Remove probe trials
    trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'laser_stimulation'] = 1

    # Get stimulation block change triggered state switches
    trials['opto_block_switch'] = np.concatenate(([False], np.diff(trials['laser_stimulation']) != 0))
    opto_block_switch_ind = np.where(trials['opto_block_switch'])[0]
    this_block_df = pd.DataFrame()
    for b, trial_ind in enumerate(opto_block_switch_ind):
            these_repeats = np.empty(len(TRIAL_BINS)-1)
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                these_repeats[tt] = np.sum(trials.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]-1,
                                                       'repeat_choice'].values)
            this_block_df = pd.concat((this_block_df, pd.DataFrame(data={
                'repeat_choice': these_repeats, 'trial_bin': trial_bin_labels, 'opto_switch': b,
                'trial_ind': np.arange(len(trial_bin_labels)),
                'opto': trials.loc[trial_ind, 'laser_stimulation']})))
    this_block_df['repeat_choice'] /= trial_bin_size

    # Get P(repeat choice)
    no_stim_trials = trials[trials['laser_stimulation'] == 0]
    stim_trials = trials[trials['laser_stimulation'] == 1]
    p_no_stim = (np.sum(no_stim_trials['choice'].values[:-1] == no_stim_trials['choice'].values[1:])
                 / (no_stim_trials.shape[0]-1))
    p_stim = (np.sum(stim_trials['choice'].values[:-1] == stim_trials['choice'].values[1:])
              / (stim_trials.shape[0]-1))

    # Add to dataframe
    this_repeats = np.concatenate(this_probe_repeat_df.groupby('trial').mean().values) * 100
    p_repeat_probe_df = pd.concat((p_repeat_probe_df, pd.DataFrame(data={
        'p_repeat': this_repeats, 'p_repeat_bl': this_repeats - np.mean(this_repeats[:PROBE_TRIALS[0]]),
        'trial': np.arange(-PROBE_TRIALS[0], PROBE_TRIALS[1]+1), 'sert-cre': sert_cre, 'subject': subject})))
    p_repeat_df = pd.concat((p_repeat_df, pd.DataFrame(index=[p_repeat_df.shape[0]+1], data={
        'p_no_stim': p_no_stim, 'p_stim': p_stim, 'subject': subject,
        'sert-cre': sert_cre})))
    this_repeats = this_probe_df.groupby('trial_ind').mean(numeric_only=True)['repeat_choice'] * 100
    p_probe_bins_df = pd.concat((p_probe_bins_df, pd.DataFrame(data={
        'p_repeat': this_repeats,
        'p_repeat_bl': this_repeats - np.mean(this_repeats.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre})))
    this_repeats = this_block_df[this_block_df['opto'] == 1].groupby('trial_ind').mean(numeric_only=True)['repeat_choice'] * 100
    p_repeat_bins_df = pd.concat((p_repeat_bins_df, pd.DataFrame(data={
        'p_repeat': this_repeats,
        'p_repeat_bl': this_repeats - np.mean(this_repeats.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 1})))
    this_repeats = this_block_df[this_block_df['opto'] == 0].groupby('trial_ind').mean(numeric_only=True)['repeat_choice'] * 100
    p_repeat_bins_df = pd.concat((p_repeat_bins_df, pd.DataFrame(data={
        'p_repeat': this_repeats,
        'p_repeat_bl': this_repeats - np.mean(this_repeats.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 0})))

# %% Statistics

# Do statistics
p_block = np.empty(trial_bin_labels.shape[0])
for t, trial in enumerate(trial_bin_labels):
    p_block[t] = ttest_1samp(p_repeat_bins_df.loc[(p_repeat_bins_df['trial'] == trial)
                                                  & (p_repeat_bins_df['sert-cre'] == 1),
                                                  'p_repeat_bl'], 0)[1]
print(p_block)

p_probe = np.empty(np.arange(-PROBE_TRIALS[0], PROBE_TRIALS[-1]+1).shape[0])
for t, trial in enumerate(np.arange(-PROBE_TRIALS[0], PROBE_TRIALS[-1]+1)):
    p_probe[t] = ttest_rel(p_repeat_probe_df.loc[(p_repeat_probe_df['trial'] == trial)
                                                 & (p_repeat_probe_df['sert-cre'] == 1),
                                                 'p_repeat'],
                           p_repeat_probe_df.loc[(p_repeat_probe_df['trial'] == -PROBE_TRIALS[0])
                                                 & (p_repeat_probe_df['sert-cre'] == 1),
                                                 'p_repeat'])[1]
    """
    p_probe[t] = ttest_1samp(p_repeat_probe_df.loc[(p_repeat_probe_df['trial'] == trial)
                                                   & (p_repeat_probe_df['sert-cre'] == 1),
                                                   'p_repeat_bl'], 0)[1]
    """
print(p_probe)



# %% Plot

p_value = ttest_rel(p_repeat_df.loc[p_repeat_df['sert-cre'] == 1, 'p_no_stim'],
                    p_repeat_df.loc[p_repeat_df['sert-cre'] == 1, 'p_stim'])[1]

colors, dpi = figure_style()
plot_colors = [colors['wt'], colors['sert']]
plot_labels = ['WT', 'SERT']
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)
for i in p_repeat_df[p_repeat_df['sert-cre'] == 1].index:
    ax1.plot([1, 2], [p_repeat_df.loc[i, 'p_no_stim'], p_repeat_df.loc[i, 'p_stim']],
             color=plot_colors[p_repeat_df.loc[i, 'sert-cre']],
             label=plot_labels[p_repeat_df.loc[i, 'sert-cre']])
ax1.set(ylabel='P[repeat choice] (%)', xticks=[1, 2], xticklabels=['No stim', 'Stim'],
        title=f'n = {len(np.unique(p_repeat_bins_df.loc[p_repeat_bins_df["sert-cre"] == 1, "subject"]))} mice',
        xlim=[0.8, 2.2], ylim=[0.6, 0.8])
#ax1.legend(frameon=False)

ax2.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_repeat_bins_df[(p_repeat_bins_df['opto'] == 1) & (p_repeat_bins_df['sert-cre'] == 1)],
             x='trial', y='p_repeat_bl', errorbar='se', err_style='bars',
             color='k', ax=ax2)
ax2.set(ylabel='P[repeat choice] (%)', xlabel='Trials since start of stimulation',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
        yticks=np.arange(-1, 6))
#handles, labels = ax1.get_legend_handles_labels()
#labels = ['No stim', '5-HT stim']
#ax2.legend(handles, labels, frameon=False, prop={'size': 5}, loc='upper left')

ax3.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_probe_bins_df[p_probe_bins_df['sert-cre'] == 1],
             x='trial', y='p_repeat_bl', errorbar='se', err_style='bars',
             color='k', ax=ax3)
ax3.set(ylabel='P[repeat choice] (%)', xlabel='Trials since single stimulation',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
        yticks=np.arange(-1, 6))
#handles, labels = ax1.get_legend_handles_labels()
#labels = ['No stim', '5-HT stim']
#ax2.legend(handles, labels, frameon=False, prop={'size': 5}, loc='upper left')

ax4.plot([-PROBE_TRIALS[0], PROBE_TRIALS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_repeat_probe_df[p_repeat_probe_df['sert-cre'] == 1], x='trial', y='p_repeat_bl',
             err_style='bars', errorbar='se', color='k', ax=ax4)
ax4.set(ylabel='P[repeat choice] (%)', xlabel='Trials since single stimulation',
        xticks=np.arange(-PROBE_TRIALS[0], PROBE_TRIALS[-1]+1))

sns.despine(trim=True)
plt.tight_layout()


