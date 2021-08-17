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
from models.expSmoothing_stimside_SE import expSmoothing_stimside_SE as exp_stimside
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials,
                                 figure_style, query_opto_sessions)
from one.api import ONE
one = ONE()

# Settings
TRIALS_AFTER_SWITCH = 20
REMOVE_OLD_FIT = True
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trial data
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        eids, stimulated='block', one=one)

    # Make array of after block switch trials
    block_switch = np.zeros(prob_left.shape)
    for k in range(prob_left.shape[0]):
        ses_length = np.sum(prob_left[k] != 0)
        trial_blocks = (prob_left[k][:ses_length] == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
        block_trans = np.append(block_trans, [trial_blocks.shape[0]])
        for s, ind in enumerate(block_trans):
            block_switch[k][ind:ind+TRIALS_AFTER_SWITCH] = 1

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_block-switch' % nickname,
                            actions, stimuli, stim_side, torch.tensor(block_switch))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'after-switch': ['no', 'yes'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
sns.lineplot(x='after-switch', y='tau', hue='sert-cre', style='subject', estimator=None,
             data=results_df.sort_values('after-switch', ascending=False),
             dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend='brief', palette=[colors['wt'], colors['sert']], lw=2, ms=8, ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 20}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='Trials after block switch', ylabel='Lenght of integration window (tau)',
        xticks=[0, 1], xticklabels=['1-20', '20+'], ylim=[0, 20])

sns.despine(trim=True, offset=10)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_block-switch_opto'))

