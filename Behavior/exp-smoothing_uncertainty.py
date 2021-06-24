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
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action_1
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action_2
from serotonin_functions import paths, criteria_opto_eids, load_exp_smoothing_trials, figure_style
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
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range_blocks'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range_blocks'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range_blocks'][:10],
                                      subjects.loc[i, 'date_range_blocks'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) == 0:
        continue

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

    # Fit model with one learning rate to get prior
    model = exp_prev_action_1('./model_fit_results/', session_uuids, '%s_uncertainty_1' % nickname,
                              actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Get uncertainty cut-offs
    priors_array = np.concatenate(priors)[np.concatenate(prob_left) != 0]
    uncertain = (priors > np.percentile(priors_array, (1/3)*100)) & (priors < np.percentile(priors_array, (2/3)*100))

    # Fit model with two learning rates: one for high and one for low uncertainty
    uncertain = (priors < 0.75) & (priors > 0.25)
    model = exp_prev_action_2('./model_fit_results/', session_uuids, '%s_uncertainty_2' % nickname,
                              actions, stimuli, stim_side, torch.tensor(uncertain))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'uncertainty': ['low', 'high'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

# %% Plot

colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
sns.lineplot(x='uncertainty', y='tau', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend='brief', palette=[colors['wt'], colors['sert']], lw=2, ms=8, ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 20}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='Uncertainty', ylabel='Length of integration window (tau)')

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_behavior'))

