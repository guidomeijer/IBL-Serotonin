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
fig_path = join(fig_path, 'Behavior', 'Models')

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
    accuracy_bayes = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                          parameter_type=POSTERIOR)['accuracy']

    # Fit previous actions model
    model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}',
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    accuracy_pa = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                       parameter_type=POSTERIOR)['accuracy']

    # Fit previous stimulus sides model
    model = exp_stimside('./model_fit_results/', session_uuids, f'{nickname}',
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    accuracy_ss = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                       parameter_type=POSTERIOR)['accuracy']

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
    accuracy_bayes = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                          parameter_type=POSTERIOR)['accuracy']

    # Fit previous actions model
    model = exp_prev_action('./model_fit_results/', session_uuids, f'{nickname}',
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    accuracy_pa = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                       parameter_type=POSTERIOR)['accuracy']

    # Fit previous stimulus sides model
    model = exp_stimside('./model_fit_results/', session_uuids, f'{nickname}',
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    accuracy_ss = model.compute_signal(signal='score', act=actions, stim=stimuli, side=stim_side,
                                       parameter_type=POSTERIOR)['accuracy']

    # Add to dataframe
    results_df = results_df.append(pd.DataFrame(data={'accuracy': [accuracy_bayes, accuracy_pa, accuracy_ss],
                                                      'model': ['optimal bayes', 'prev actions', 'stim side'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'stim': 0, 'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi)
colors_id = [colors['wt'], colors['sert']]

for i, subject in enumerate(results_df['subject'].unique()):
    ax1.plot([1, 2, 3],
             [results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'optimal bayes')
                             & (results_df['stim'] == 0)), 'accuracy'],
              results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'stim side')
                              & (results_df['stim'] == 0)), 'accuracy'],
              results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'prev actions')
                              & (results_df['stim'] == 0)), 'accuracy']],
             color=colors_id[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
ax1.set(ylim=[0.55, 0.75], ylabel='Model accuracy', xticks=[1, 2, 3], title='Non-stimulated sessions',
        xticklabels=['Bayes\noptimal', 'Stim.\nside', 'Prev.\nactions'])


for i, subject in enumerate(results_df['subject'].unique()):
   ax2.plot([1, 2, 3],
             [results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'optimal bayes')
                             & (results_df['stim'] == 1)), 'accuracy'],
              results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'stim side')
                              & (results_df['stim'] == 1)), 'accuracy'],
              results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'prev actions')
                              & (results_df['stim'] == 1)), 'accuracy']],
             color=colors_id[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
ax2.set(ylim=[0.55, 0.75], ylabel='Model accuracy', xticks=[1, 2, 3], title='Stimulated sessions',
        xticklabels=['Bayes\noptimal', 'Stim.\nside', 'Prev.\nactions'])

for i, subject in enumerate(results_df['subject'].unique()):
    ax3.plot([1, 2],
             [(results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'prev actions')
                               & (results_df['stim'] == 0)), 'accuracy'].values
               - results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'optimal bayes')
                                 & (results_df['stim'] == 0)), 'accuracy'].values),
              (results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'prev actions')
                               & (results_df['stim'] == 1)), 'accuracy'].values
               - results_df.loc[((results_df['subject'] == subject) & (results_df['model'] == 'optimal bayes')
                                 & (results_df['stim'] == 1)), 'accuracy'].values)],
             color=colors_id[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
ax3.set(xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylabel='prev actions - optimal bayes',
        ylim=[0, 0.08])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Model accuracy comparison'))