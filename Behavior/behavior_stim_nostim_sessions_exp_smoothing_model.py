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
    if subjects.loc[i, 'date_range_blocks'] == 'all':
        eids, details = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                   details=True)
    elif subjects.loc[i, 'date_range_blocks'] == 'none':
        continue
    else:
        eids, details = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                                   date_range=[subjects.loc[i, 'date_range_blocks'][:10],
                                               subjects.loc[i, 'date_range_blocks'][11:]], details=True)

    stim_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in details]]
    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=400, one=one)
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_stim_ses' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'tau_pa': 1/param_prevaction[0], 'sessions': 'stim',
        'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Pre unstimulated sessions
    eids, details = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                               details=True)
    no_stim_dates = [datetime.strptime(i, '%Y-%m-%d') for i in [j['start_time'][:10] for j in details]]
    pre_dates = [i for i in no_stim_dates if i < np.min(stim_dates)]
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                      date_range=[str(pre_dates[4])[:10], str(pre_dates[0])[:10]])
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_pre_ses' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'tau_pa': 1/param_prevaction[0], 'sessions': 'pre',
        'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

    # Post unstimulated sessions
    post_dates = [i for i in no_stim_dates if i > np.max(stim_dates)]
    if len(post_dates) > 0:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_biasedChoiceWorld',
                          date_range=[str(post_dates[-1])[:10], str(post_dates[0])[:10]])
        actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)
        model = exp_prev_action('./model_fit_results/', session_uuids, '%s_post_ses' % nickname,
                                actions, stimuli, stim_side)
        model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
        param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'tau_pa': 1/param_prevaction[0], 'sessions': 'post',
        'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

# %% Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x='sessions', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False,
             legend=False, ax=ax1)
ax1.set(ylabel='Tau', xticks=[0, 1],
        xticklabels=['Non-stimulated', 'Stimulated'],
        xlabel='')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'stim_vs_nonstim_ses_opto_behavior'))

