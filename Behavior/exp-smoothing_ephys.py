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
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials, figure_style,
                                 load_subjects)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 16
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Ephys')

subjects = load_subjects()

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_ephysChoiceWorld')
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue

    # Get trial data
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        eids, stimulated='block', one=one)

    # Fit models
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                            actions, stimuli, stim_side, torch.tensor(stimulated))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors_prevaction = model.compute_signal(signal='prior', act=actions, stim=stimuli,
                                             side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'opto_stim': ['no stim', 'stim'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))


    # Add prior around block switch non-stimulated
    for k in range(len(priors_prevaction)):
        transitions = np.array(np.where(np.diff(prob_left[k]) != 0)[0])[:-1] + 1
        for t, trans in enumerate(transitions):
            if trans >= PRE_TRIALS:
                if stimulated[k][trans] == 1:
                    opto = 'stim'
                elif stimulated[k][trans] == 0:
                    opto = 'no stim'
                block_switches = block_switches.append(pd.DataFrame(data={
                            'prior_prevaction': priors_prevaction[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                            'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                            'change_to': prob_left[k][trans],
                            'opto': opto,
                            'sert_cre': subjects.loc[i, 'sert-cre'],
                            'subject': nickname}))
    block_switches = block_switches.reset_index(drop=True)

    # Plot for this animal
    colors, dpi = figure_style()
    f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
    sns.lineplot(x='trial', y='prior_prevaction', data=block_switches[block_switches['subject'] == nickname],
             hue='change_to', style='opto', palette='colorblind', ax=ax1, ci=68)
    #plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
    ax1.legend(handles, labels, frameon=False, bbox_to_anchor=(1, 1))
    ax1.set(ylabel='Prior', xlabel='Trials relative to block switch', ylim=[0, 1],
            title='Tau stim: %.2f, Tau no stim: %.2f' % (1/param_prevaction[1], 1/param_prevaction[0]))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_model_prevaction' % nickname))

results_df = results_df.reset_index(drop=True)

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi)

sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Length of integration window (tau)',
        title='Previous actions model')

sns.lineplot(x='trial', y='prior_prevaction', data=block_switches[block_switches['sert_cre'] == 1],
             hue='change_to', style='opto', palette='colorblind', ax=ax2, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax2.legend(handles, labels, frameon=False)
ax2.set(ylabel='Prior', xlabel='Trials relative to block switch', title='Sert-Cre', ylim=[0, 1])

sns.lineplot(x='trial', y='prior_prevaction', data=block_switches[block_switches['sert_cre'] == 0],
             hue='change_to', style='opto', palette='colorblind', ax=ax3, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'No stim', 'Stim']
ax3.legend(handles, labels, frameon=False)
ax3.set(ylabel='Prior', xlabel='Trials relative to block switch', title='WT control', ylim=[0, 1])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_behavior'))