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
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from serotonin_functions import paths, criteria_opto_eids, load_exp_smoothing_trials
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 16
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Stimulated sessions
    if subjects.loc[i, 'date_range_block'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range_block'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range'][:10], subjects.loc[i, 'date_range'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=400, one=one)
    if len(eids) == 0:
        continue

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit models
    """
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_stim_ses' % nickname,
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    """

    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_stim_ses' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'tau_pa': 1/param_prevaction[0],
        'opto_stim': True, 'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Unstimulated sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld')
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[-10:]  # select last 10 sessions

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit models
    """
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_no_stim_ses' % nickname,
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    """

    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_no_stim_ses' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'tau_pa': 1/param_prevaction[0],
        'opto_stim': False, 'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

# %% Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(ylabel='Tau', xticks=[0, 1],
        xticklabels=['Non-stimulated', 'Stimulated'],
        xlabel='')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim_vs_nonstim_ses_opto_behavior'))

