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
from matplotlib.patches import Rectangle
from models.expSmoothing_stimside_SE import expSmoothing_stimside_SE as exp_stimside
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials, figure_style,
                                 query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
TRIALS_AFTER_SWITCH = 20
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
STIM = 'block'
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects()

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, include_ephys=True, one=one)
    eids = behavioral_criterion(eids, verbose=False, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
        eids, stimulated=STIM, patch_old_opto=False, one=one)

    # Make array of after block switch trials
    block_switch = np.zeros(prob_left.shape)
    for k in range(prob_left.shape[0]):
        ses_length = np.sum(prob_left[k] != 0)
        trial_blocks = (prob_left[k][:ses_length] == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
        block_trans = np.append(block_trans, [trial_blocks.shape[0]])
        for s, ind in enumerate(block_trans):
            block_switch[k][ind:ind+TRIALS_AFTER_SWITCH] = 1

    early_block = (block_switch == 1).astype(int)

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_late_laser_block' % nickname,
                            actions, stimuli, stim_side, torch.tensor(early_block))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors_prevaction = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Add to df
    results_df = results_df.append(pd.DataFrame(data={'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'early_block': ['early', 'late'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi, sharey=False)


sns.lineplot(x='early_block', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             hue_order=[1, 0], palette=[colors['sert'], colors['wt']],
             legend=False, ax=ax1)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='Previous actions',
        xticks=[0, 1], xticklabels=['Early block', 'Late block'], ylim=[0, 30])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'exp-smoothing_early_early_block.jpg'), dpi=600)
