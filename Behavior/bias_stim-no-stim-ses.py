#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 load_subjects, fit_psychfunc, query_opto_sessions)
from one.api import ONE
one = ONE()

# Settings
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Psychometrics')

subjects = load_subjects()
subjects = subjects[subjects['subject'] != 'ZFM-02602']
subjects = subjects[subjects['subject'] != 'ZFM-02180']
subjects = subjects[subjects['subject'] != 'ZFM-01867']
subjects = subjects.reset_index()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}')
    """
    # Query opto sessions
    eids = query_opto_sessions(nickname)

    # Apply behavioral criterion
    #eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, one=one)
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if trials.shape[0] == 0:
        continue

    # Get fit parameters
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left_stim = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                                      these_trials.groupby('signed_contrast').mean()['right_choice'])

    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right_stim = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                                       these_trials.groupby('signed_contrast').mean()['right_choice'])

    # Query non-opto sessions
    sessions = one.alyx.rest('sessions', 'list', subject=nickname,
                             task_protocol='_iblrig_tasks_biasedChoiceWorld',
                             project='serotonin_inference')
    eids = [sess['url'][-36:] for sess in sessions]

    # Apply behavioral criterion
    #eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_trials(eid, laser_stimulation=False, one=one)
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
        except:
            pass
    if trials.shape[0] == 0:
        continue

    # Get fit parameters
    these_trials = trials[trials['probabilityLeft'] == 0.8]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_left_no_stim = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                                      these_trials.groupby('signed_contrast').mean()['right_choice'])

    these_trials = trials[trials['probabilityLeft'] == 0.2]
    stim_levels = np.sort(these_trials['signed_contrast'].unique())
    pars_right_no_stim = fit_psychfunc(stim_levels, these_trials.groupby('signed_contrast').size(),
                                       these_trials.groupby('signed_contrast').mean()['right_choice'])


    results_df = results_df.append(pd.DataFrame(index=[len(results_df)+1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'bias_left_no_stim': pars_left_no_stim[0],
        'bias_right_no_stim': pars_right_no_stim[0], 'bias_no_stim': np.abs(pars_left_no_stim[0] - pars_right_no_stim[0]),
        'bias_left_stim': pars_left_stim[0], 'bias_right_stim': pars_right_stim[0],
        'bias_stim': np.abs(pars_left_stim[0] - pars_right_stim[0])}))
    """

    sessions = one.alyx.rest('sessions', 'list', subject=nickname,
                             task_protocol='biased',
                             project='serotonin_inference')
    eids = [sess['url'][-36:] for sess in sessions]

    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, one=one)
        except:
            continue
        details = one.get_details(eid)
        bias = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['signed_contrast'] == 0)].mean()
                      - trials[(trials['probabilityLeft'] == 0.2)
                               & (trials['signed_contrast'] == 0)].mean())['right_choice']
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'bias': bias, 'stim': int('opto' in details['task_protocol']),
            'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))


# %% Plot
colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3.5), dpi=dpi)

for i, subject in enumerate(results_df['subject'].unique()):
    if results_df[(results_df['subject'] == subject) & (results_df['stim'] == 1)].shape[0] == 0:
        continue
    ax1.plot([0, 1], [results_df.loc[(results_df['subject'] == subject) & (results_df['stim'] == 0), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == subject) & (results_df['stim'] == 1), 'bias'].mean()],
             color=colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
ax1.set(ylabel='Bias', xticks=[0, 1], xticklabels=['No stim', 'Stim'])

for i, subject in enumerate(results_df['subject'].unique()):
    if results_df[(results_df['subject'] == subject) & (results_df['stim'] == 1)].shape[0] == 0:
        continue
    ax2.plot([0, 1], [results_df.loc[(results_df['subject'] == subject) & (results_df['stim'] == 0), 'bias'].mean(),
                      results_df.loc[(results_df['subject'] == subject) & (results_df['stim'] == 1), 'bias'].mean()],
             color=colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
ax2.set(ylabel='Bias', xticks=[0, 1], xticklabels=['No stim', 'Stim'])

plt.tight_layout()
plt.savefig(join(fig_path, 'stim-no-stim-bias'))

