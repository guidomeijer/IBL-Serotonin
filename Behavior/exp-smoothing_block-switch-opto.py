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
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials,
                                 figure_style, query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
TRIALS_AFTER_SWITCH = 20
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects(behavior=True)

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    #eids = query_opto_sessions(nickname, one=one)
    sessions = one.alyx.rest('sessions', 'list', subject=nickname,
                             task_protocol='biased',
                             project='serotonin_inference')
    eids = [sess['url'][-36:] for sess in sessions]
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(eids, one=one)

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
f, ax1 = plt.subplots(1, 1, figsize=(2, 2.1), dpi=dpi)

colors = [colors['wt'], colors['sert']]
for i, subject in enumerate(results_df['subject']):
    sert_cre = results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]
    ax1.plot([1, 2], results_df.loc[(results_df['subject'] == subject), 'tau'],
             color = colors[sert_cre], marker='o', ms=2)

ax1.set(xlabel='Trials after block switch', ylabel='Integration window (tau)',
        xticks=[1, 2], xticklabels=['1-20', '21+'], ylim=[0, 15])

sns.despine(trim=True, offset=10)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_block-switch_no_opto'))

