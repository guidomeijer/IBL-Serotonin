#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                                 fit_psychfunc, figure_style, query_opto_sessions, get_bias,
                                 load_subjects)
from one.api import ONE
one = ONE()

# Settings
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')
subjects = load_subjects()

bias_df, lapse_df, psy_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

    # Exclude the first opto sessions
    #eids = eids[:-1]

    # Apply behavioral criterion
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:

            if subjects.loc[i, 'sert-cre'] == 1:
                these_trials = load_trials(eid, laser_stimulation=True, one=one)
            else:
                these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            """
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            """
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if len(trials) == 0:
        continue

    # Get fit parameters
    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 0)
                          & (trials['probe_trial'] == 0)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 0)
                          & (trials['probe_trial'] == 0)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 0, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.8) & (trials['laser_stimulation'] == 1)
                          & (trials['probe_trial'] == 0)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.8,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

    these_trials = trials[(trials['probabilityLeft'] == 0.2) & (trials['laser_stimulation'] == 1)
                          & (trials['probe_trial'] == 0)]
    pars = fit_psychfunc(np.sort(these_trials['signed_contrast'].unique()),
                         these_trials.groupby('signed_contrast').size(),
                         these_trials.groupby('signed_contrast').mean()['right_choice'])
    psy_df = psy_df.append(pd.DataFrame(index=[len(psy_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'opto_stim': 1, 'prob_left': 0.2,
        'bias': pars[0], 'threshold': pars[1], 'lapse_l': pars[2], 'lapse_r': pars[3]}))

# %% Plot
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(6.5, 3.5), dpi=dpi)

for i, subject in enumerate(psy_df['subject']):
    ax1.plot([0, 1],
             [psy_df.loc[((psy_df['subject'] == subject) & (psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)), 'bias'],
              psy_df.loc[((psy_df['subject'] == subject) & (psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)), 'bias']],
             color = colors[int(psy_df.loc[psy_df['subject'] == subject, 'sert-cre'].unique())],
             marker='o', ms=2)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Bias left')

for i, subject in enumerate(psy_df['subject']):
    ax2.plot([0, 1],
             [psy_df.loc[((psy_df['subject'] == subject) & (psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)), 'bias'],
              psy_df.loc[((psy_df['subject'] == subject) & (psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)), 'bias']],
             color = colors[int(psy_df.loc[psy_df['subject'] == subject, 'sert-cre'].unique())],
             marker='o', ms=2)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Bias right')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'summary_psycurve.pdf'))

# %%
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=dpi)
delta_lapse_l_l_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                & (psy_df['sert-cre'] == 1), 'lapse_l'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                              & (psy_df['sert-cre'] == 1), 'lapse_l'].values)
delta_lapse_l_r_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                & (psy_df['sert-cre'] == 1), 'lapse_l'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                              & (psy_df['sert-cre'] == 1), 'lapse_l'].values)
delta_lapse_r_l_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                & (psy_df['sert-cre'] == 1), 'lapse_r'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                  & (psy_df['sert-cre'] == 1), 'lapse_r'].values)
delta_lapse_r_r_s = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                    & (psy_df['sert-cre'] == 1), 'lapse_r'].values
                     - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                  & (psy_df['sert-cre'] == 1), 'lapse_r'].values)
delta_lapse_l_l_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                 & (psy_df['sert-cre'] == 0), 'lapse_l'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                  & (psy_df['sert-cre'] == 0), 'lapse_l'].values)
delta_lapse_l_r_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                 & (psy_df['sert-cre'] == 0), 'lapse_l'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                   & (psy_df['sert-cre'] == 0), 'lapse_l'].values)
delta_lapse_r_l_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.8)
                                 & (psy_df['sert-cre'] == 0), 'lapse_r'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.8)
                                   & (psy_df['sert-cre'] == 0), 'lapse_r'].values)
delta_lapse_r_r_wt = (psy_df.loc[(psy_df['opto_stim'] == 1) & (psy_df['prob_left'] == 0.2)
                                 & (psy_df['sert-cre'] == 0), 'lapse_r'].values
                      - psy_df.loc[(psy_df['opto_stim'] == 0) & (psy_df['prob_left'] == 0.2)
                                   & (psy_df['sert-cre'] == 0), 'lapse_r'].values)

ax1.plot([-.5, 1.5], [0, 0], ls='--', color=[.5, .5, .5], lw=2)
ax1.plot(np.zeros(len(delta_lapse_l_l_s)), delta_lapse_l_l_s, 'o', color=colors[0])
ax1.plot(np.zeros(len(delta_lapse_l_l_wt)), delta_lapse_l_l_wt, 'o', color=colors[1])
ax1.plot(np.ones(len(delta_lapse_r_l_s)), delta_lapse_r_l_s, 'o', color=colors[0])
ax1.plot(np.ones(len(delta_lapse_r_l_wt)), delta_lapse_r_l_wt, 'o', color=colors[1])
ax1.set(xticks=[0, 1], xticklabels=['L', 'R'], ylabel='delta lapse rate \n (stim minus non-stim)',
        ylim=[-.1, .1], title='80:20 blocks')

ax2.plot([-.5, 1.5], [0, 0], ls='--', color=[.5, .5, .5], lw=2)
ax2.plot(np.zeros(len(delta_lapse_l_r_s)), delta_lapse_l_r_s, 'o', color=colors[0])
ax2.plot(np.zeros(len(delta_lapse_l_r_wt)), delta_lapse_l_r_wt, 'o', color=colors[1])
ax2.plot(np.ones(len(delta_lapse_r_r_s)), delta_lapse_r_r_s, 'o', color=colors[0])
ax2.plot(np.ones(len(delta_lapse_r_r_wt)), delta_lapse_r_r_wt, 'o', color=colors[1])
ax2.set(xticks=[0, 1], xticklabels=['L', 'R'], ylabel='delta lapse rate \n (stim minus non-stim)',
        ylim=[-.1, .1], title='20:80 blocks')
plt.tight_layout()
"""