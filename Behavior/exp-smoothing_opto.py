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
from serotonin_functions import (paths, behavioral_criterion, load_exp_smoothing_trials, figure_style,
                                 query_opto_sessions, load_subjects)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
STIM = 'all'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects(behavior=True)

results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    #eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    """
    if subjects.loc[i, 'sert-cre'] == 1:
        actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=True, one=one)
    else:
        actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=False, one=one)
    """
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        eids, stimulated=STIM, patch_old_opto=True, one=one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_%s' % (nickname, STIM),
                            actions, stimuli, stim_side, torch.tensor(stimulated))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Fit model
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_%s' % (nickname, STIM),
                         actions, stimuli, stim_side, torch.tensor(stimulated))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_stimside = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Add to df
    results_df = results_df.append(pd.DataFrame(data={'tau_ss': [1/param_stimside[0], 1/param_stimside[1]],
                                                      'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'opto_stim': ['no stim', 'stim'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'expression': subjects.loc[i, 'expression'],
                                                      'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi, sharey=False)

for i, subject in enumerate(results_df['subject']):
    ax1.plot([1, 2], results_df.loc[(results_df['subject'] == subject), 'tau_pa'],
             color = colors[results_df.loc[results_df['subject'] == subject, 'expression'].unique()[0]],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='Previous actions',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 15])

for i, subject in enumerate(results_df['subject']):
    ax2.plot([1, 2], results_df.loc[(results_df['subject'] == subject), 'tau_ss'],
             color = colors[results_df.loc[results_df['subject'] == subject, 'expression'].unique()[0]],
             marker='o', ms=2)
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax2.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax2.set(xlabel='', ylabel='Length of integration window (tau)', title='Stimulus sides',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 25])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'exp-smoothing_opto_{STIM}'))
