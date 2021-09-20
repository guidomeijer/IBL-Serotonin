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
import torch
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

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_block' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Fit model
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_block' % nickname,
                         actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau': [1/param_prevaction[0], 1/param_stimside[1]],
                                                      'model': ['prev actions', 'stim side'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
sns.stripplot(x='sert-cre', y='tau', data=results_df[results_df['model'] == 'prev actions'], s=4,
              palette=[colors['wt'], colors['sert']], ax=ax1)
ax1.set(ylabel='Tau', xlabel='', xticks=[0, 1], xticklabels=['WT', 'Sert-Cre'], ylim=[2, 8],
        title='Previous actions')

sns.stripplot(x='sert-cre', y='tau', data=results_df[results_df['model'] == 'stim side'], s=4,
              palette=[colors['wt'], colors['sert']], ax=ax2)
ax2.set(ylabel='Tau', xlabel='', xticks=[0, 1], xticklabels=['WT', 'Sert-Cre'],
        title='Stimulus sides')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'exp_smoothing_whole_session'))
