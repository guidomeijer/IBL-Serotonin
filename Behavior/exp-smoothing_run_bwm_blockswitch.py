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
from scipy.stats import wilcoxon
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import paths, query_bwm_sessions, load_exp_smoothing_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
TRIALS_AFTER_SWITCH = 20
REMOVE_OLD_FIT = True
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT')
save_path = join(save_path, '5HT')

# Query brain wide map sessions
eids, _, subjects = query_bwm_sessions(selection='aligned-behavior', return_subjects=True, one=one)

results_df = pd.DataFrame()
for i, nickname in enumerate(np.unique(subjects)):
    print(f'Processing subject {nickname} [{i+1} of {len(np.unique(subjects))}]')

    # Get trial data
    these_eids = eids[subjects == nickname]
    actions, stimuli, stim_side, prob_left, session_uuids = load_exp_smoothing_trials(these_eids, one=one)
    if len(session_uuids) == 0:
        continue

    # Make array of after block switch trials
    block_switch = np.zeros(prob_left.shape)
    for k in range(prob_left.shape[0]):
        ses_length = np.sum(prob_left[k] != 0)
        trial_blocks = (prob_left[k][:ses_length] == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
        block_trans = np.append(block_trans, [trial_blocks.shape[0]])
        for s, ind in enumerate(block_trans):
            block_switch[k][ind:ind+TRIALS_AFTER_SWITCH] = 1

    # Fit model with two learning rates: one for high and one for low uncertainty
    if len(block_switch.shape) == 1:
        block_switch = [block_switch]
    model = exp_prev_action('./model_fit_results/brain_wide_map', session_uuids, '%s' % nickname,
                              actions, stimuli, stim_side, torch.tensor(block_switch))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'after-switch': ['no', 'yes'],
                                                      'subject': nickname}))
    results_df.to_csv(join(save_path, 'brain_wide_map_exp_smoothing'))
results_df.to_csv(join(save_path, 'brain_wide_map_exp_smoothing'))

# %% Plot

# Test
_, p = wilcoxon(results_df[results_df['after-switch'] == 'yes']['tau'].values,
                results_df[results_df['after-switch'] == 'no']['tau'].values)

colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(4, 5), dpi=150)
sns.lineplot(x='after-switch', y='tau', style='subject', estimator=None, lw=1, ms=0,
             data=results_df.sort_values('after-switch', ascending=False), color=[.7, .7, .7],
             dashes=False, markers=['o']*int(results_df.shape[0]/2), legend=False, ax=ax1)
plt.errorbar([0, 1], [results_df[results_df['after-switch'] == 'yes'].mean()['tau'],
                      results_df[results_df['after-switch'] == 'no'].mean()['tau']],
             yerr=[results_df[results_df['after-switch'] == 'yes'].sem()['tau'],
                   results_df[results_df['after-switch'] == 'no'].sem()['tau']], marker='o', lw=3)
ax1.set(xlabel='Trials after block switch', ylabel='Length of integration window (tau)',
        xticks=[0, 1], xticklabels=['1-20', '21+'], ylim=[0, 30])

sns.despine(trim=True, offset=10)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_bwm_after-switch'))

