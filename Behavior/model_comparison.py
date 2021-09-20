#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

Model comparisons are done by fitting the different models on all the sessions except the last one.
Then the model performance is quantified as it's peformance on the last held-out sessions.

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import torch
from models.optimalBayesian import optimal_Bayesian as opt_bayes
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials,
                                 figure_style, query_opto_sessions)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 16
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Stimulated sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)
    if len(eids) < 2:
        continue
    if len(eids) > 10:
        eids = eids[:10]
    details = [one.get_details(i) for i in eids]
    stim_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in details]]

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit optimal bayesian model
    model = opt_bayes('./model_fit_results/', session_uuids, f'{nickname}',
                      actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    #accuracy_bayes = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']
    accuracy_bayes = np.nan

    # Fit previous actions model
    model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}',
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    accuracy_pa = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']

    # Fit previous stimulus sides model
    model = exp_stimside('./model_fit_results/', session_uuids, f'{nickname}',
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    accuracy_ss = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']

    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={'accuracy': [accuracy_bayes, accuracy_pa, accuracy_ss],
                                                      'model': ['optimal bayes', 'prev actions', 'stim side'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'stim': 1, 'subject': nickname}))

    # Non stimulated sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld')
    eids = behavioral_criterion(eids, one=one)
    details = [one.get_details(i) for i in eids]
    no_stim_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in details]]
    pre_dates = [i for i in no_stim_dates if i < np.min(stim_dates)]
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                      date_range=[str(pre_dates[4])[:10], str(pre_dates[0])[:10]])

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit optimal bayesian model
    model = opt_bayes('./model_fit_results/', session_uuids, f'{nickname}',
                      actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    #accuracy_bayes = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']
    accuracy_bayes = np.nan

    # Fit previous actions model
    model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}',
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    accuracy_pa = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']

    # Fit previous stimulus sides model
    model = exp_stimside('./model_fit_results/', session_uuids, f'{nickname}',
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    accuracy_ss = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side)['accuracy']

    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={'accuracy': [accuracy_bayes, accuracy_pa, accuracy_ss],
                                                      'model': ['optimal bayes', 'prev actions', 'stim side'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'stim': 0, 'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=dpi)
sns.stripplot(x='sert-cre', y='tau', data=results_df[results_df['model'] == 'prev actions'], s=8,
              palette=[colors['wt'], colors['sert']], ax=ax1)
ax1.set(ylabel='Tau', xlabel='', xticks=[0, 1], xticklabels=['WT', 'Sert-Cre'],
        title='Previous actions')

sns.stripplot(x='sert-cre', y='tau', data=results_df[results_df['model'] == 'stim side'], s=8,
              palette=[colors['wt'], colors['sert']], ax=ax2)
ax2.set(ylabel='Tau', xlabel='', xticks=[0, 1], xticklabels=['WT', 'Sert-Cre'],
        title='Stimulus sides')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'exp_smoothing_whole_session'))
