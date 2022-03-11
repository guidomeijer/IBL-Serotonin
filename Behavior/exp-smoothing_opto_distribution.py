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
PRE_TRIALS = 5
POST_TRIALS = 20
PLOT_EXAMPLES = True
REMOVE_OLD_FIT = False
POSTERIOR = 'all'
STIM = 'block'
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects(behavior=True)

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, one=one)
    if len(eids) == 0:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    if subjects.loc[i, 'sert-cre'] == 1:
        actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=True, one=one)
    else:
        actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
            eids, stimulated=STIM, patch_old_opto=False, one=one)

    # Fit model
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_%s' % (nickname, STIM),
                            actions, stimuli, stim_side, torch.tensor(stim_trials))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)

    plt.figure()
    plt.hist(np.concatenate(1/param_prevaction[1001:, :, 0]), bins=50)
    plt.hist(np.concatenate(1/param_prevaction[1001:, :, 1]), bins=50)
    plt.title(nickname)


    # Add to df
    results_df = pd.concat((results_df, pd.DataFrame(data={'tau_pa': np.concatenate(param_prevaction[1001:, :, 0]),
                                                           'opto_stim': 'no stim', 'sert-cre': subjects.loc[i, 'sert-cre'],
                                                           'subject': nickname})))
    results_df = pd.concat((results_df, pd.DataFrame(data={'tau_pa': np.concatenate(param_prevaction[1001:, :, 1]),
                                                           'opto_stim': 'stim', 'sert-cre': subjects.loc[i, 'sert-cre'],
                                                           'subject': nickname})))



# %% Plot

colors, dpi = figure_style()
colors = [colors['wt'], colors['sert']]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi, sharey=False)

for i, subject in enumerate(results_df['subject']):
    ax1.plot([1, 2], results_df.loc[(results_df['subject'] == subject), 'tau_pa'],
             color = colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='Previous actions',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 15])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'exp-smoothing_opto_{STIM}'), dpi=300)
