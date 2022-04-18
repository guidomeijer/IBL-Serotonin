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
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, behavioral_criterion, load_trials, figure_style,
                                 query_opto_sessions, load_subjects, fit_glm)
from one.api import ONE
one = ONE()
log_reg = LogisticRegression(random_state=42)

# Settings
PAST_CHOICE_WIN = 10  # trials
MIN_TRIALS = 3  # for estimating choice bias
subjects = load_subjects(behavior=True)
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'ModelAgnostic')

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)

    # Get trials DataFrame
    all_trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            """
            if subjects.loc[i, 'sert-cre'] == 1:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=True, one=one)
            else:
                trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
            """
            trials = load_trials(eid, laser_stimulation=True, one=one)
        except:
            continue

        for t in range(PAST_CHOICE_WIN+1, trials.shape[0]):
            trials_slice = trials[t-(PAST_CHOICE_WIN+1):t-1]

            # reward bias opto
            stim_trials = trials_slice[trials_slice['laser_stimulation'] == 1]
            if stim_trials.shape[0] >= MIN_TRIALS:
                trials.loc[t, 'past_choices_opto'] = np.sum(stim_trials['choice'])

            # reward bias no opto
            no_stim_trials = trials_slice[trials_slice['laser_stimulation'] == 0]
            if no_stim_trials.shape[0] >= MIN_TRIALS:
                trials.loc[t, 'past_choices_no_opto'] = np.sum(no_stim_trials['choice'])

            # all choices
            trials.loc[t, 'past_choices'] = np.sum(trials_slice['choice'])

        # Add to trial dataframe
        all_trials = pd.concat((all_trials, trials), ignore_index=True)

    # Fit logistic regression
    """
    trials_slice = all_trials.loc[~all_trials['past_choices_opto'].isnull()]
    log_reg.fit(trials_slice['past_choices_opto'].values.reshape(-1, 1), trials_slice['choice'].values)
    score_opto = log_reg.score(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)
    coef_opto = log_reg.coef_[0][0]

    trials_slice = all_trials.loc[~all_trials['past_choices_no_opto'].isnull()]
    log_reg.fit(trials_slice['past_choices_no_opto'].values.reshape(-1, 1), trials_slice['choice'].values)
    score_no_opto = log_reg.score(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)
    coef_no_opto = log_reg.coef_[0][0]
    """

    trials_slice = all_trials.loc[~all_trials['past_choices'].isnull()]
    log_reg.fit(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)
    coef_all = log_reg.coef_[0][0]
    score_all = log_reg.score(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)

    trials_slice = all_trials.loc[(all_trials['laser_stimulation'] == 1) & (~all_trials['past_choices'].isnull())]
    log_reg.fit(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)
    coef_opto = log_reg.coef_[0][0]
    score_opto = log_reg.score(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)

    trials_slice = all_trials.loc[(all_trials['laser_stimulation'] == 0) & (~all_trials['past_choices'].isnull())]
    log_reg.fit(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)
    coef_no_opto = log_reg.coef_[0][0]
    score_no_opto = log_reg.score(trials_slice['past_choices'].values.reshape(-1, 1), trials_slice['choice'].values)

    results_df = pd.concat((results_df, pd.DataFrame(data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'],
        'coef': [coef_opto, coef_no_opto], 'score': [score_opto, score_no_opto],
        'opto': [1, 0]})), ignore_index=True)

# %% Plot

plot_df = results_df.copy()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] = 'Sert'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] = 'WT'
plot_df.loc[plot_df['opto'] == 1, 'opto'] = 'Stim'
plot_df.loc[plot_df['opto'] == 0, 'opto'] = 'No stim'

colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.lineplot(x='opto', y='coef', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax1)
ax1.legend(frameon=False)
ax1.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Coefficient')

sns.lineplot(x='opto', y='score', data=plot_df, hue='sert-cre', estimator=None, units='subject',
             palette=[colors['sert'], colors['wt']], legend='brief', dashes=False,
             markers=['o']*int(plot_df.shape[0]/2), ax=ax2)
ax2.legend(frameon=False)
ax2.set(xlabel='', xticks=[0, 1], xticklabels=['No stim', 'Stim'], ylabel='Coefficient')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'log_reg_opto'), dpi=300)

