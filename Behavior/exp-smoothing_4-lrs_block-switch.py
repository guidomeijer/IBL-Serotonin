#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models.expSmoothing_prevAction_4lr import expSmoothing_prevAction_4lr as exp_prev_action
from serotonin_functions import (paths, load_exp_smoothing_trials, figure_style, load_subjects,
                                 query_opto_sessions, behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
TRIALS_AFTER_SWITCH = 20
POSTERIOR = 'posterior_mean'
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects()

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

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

    stim_block = np.zeros(stimulated.shape)
    stim_block[(stimulated == 0) & (block_switch == 0)] = 0
    stim_block[(stimulated == 0) & (block_switch == 1)] = 1
    stim_block[(stimulated == 1) & (block_switch == 0)] = 2
    stim_block[(stimulated == 1) & (block_switch == 1)] = 3
    stim_block = torch.tensor(stim_block.astype(int))

    if len(session_uuids) == 0:
        continue

    # Fit models
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_blockswitch_opto' % nickname,
                            actions, stimuli, stim_side, torch.tensor(stim_block))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)

    output_prevaction = model.compute_signal(signal=['prior', 'score'], act=actions, stim=stimuli, side=stim_side)
    priors_prevaction = output_prevaction['prior']
    results_df = results_df.append(pd.DataFrame(data={
        'tau_pa': 1/param_prevaction[:4], 'opto_stim': [0, 0, 1, 1], 'subject': nickname,
        'block_switch': ['late', 'early', 'late', 'early'], 'sert-cre': subjects.loc[i, 'sert-cre']}))
results_df = results_df.reset_index(drop=True)
# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.5, 3.5), dpi=dpi)

sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df[results_df['block_switch'] == 'early'],
             dashes=False, markers=['o']*int(results_df.shape[0]/4),
             hue_order=[1, 0], palette=[colors['sert'], colors['wt']],
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Length of integration window (tau)', xticks=[0, 1],
        xticklabels=['No stim', 'Stim'],
        title=f'Early block (<{TRIALS_AFTER_SWITCH} trials)')

sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df[results_df['block_switch'] == 'late'],
             dashes=False, markers=['o']*int(results_df.shape[0]/4),
             hue_order=[1, 0], palette=[colors['sert'], colors['wt']],
             legend=False, ax=ax2)
ax2.set(xlabel='', ylabel='Length of integration window (tau)', xticks=[0, 1],
        xticklabels=['No stim', 'Stim'],
        title=f'Late block (>{TRIALS_AFTER_SWITCH} trials)')

sns.lineplot(x='block_switch', y='tau_pa', hue='opto_stim', style='subject', estimator=None,
             data=results_df[results_df['sert-cre'] == 1],
             dashes=False, markers=['o']*int(results_df.shape[0]/4),
             hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']],
             legend=False, ax=ax3)
ax3.set(xlabel='', ylabel='Length of integration window (tau)', title='SERT')

sns.lineplot(x='block_switch', y='tau_pa', hue='opto_stim', style='subject', estimator=None,
             data=results_df[results_df['sert-cre'] == 0],
             dashes=False, markers=['o']*int(results_df.shape[0]/4),
             hue_order=[1, 0], palette=[colors['stim'], colors['no-stim']],
             legend=False, ax=ax4)
ax4.set(xlabel='', ylabel='Length of integration window (tau)', title='WT')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_block-switch.jpg'), dpi=600)
