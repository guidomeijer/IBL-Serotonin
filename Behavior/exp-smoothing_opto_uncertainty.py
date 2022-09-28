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
from models.expSmoothing_prevAction import expSmoothing_prevAction as prev_action_1lr
from models.expSmoothing_prevAction_4lr import expSmoothing_prevAction_4lr as prev_action_4lr
from serotonin_functions import (paths, load_exp_smoothing_trials, figure_style, query_opto_sessions,
                                 load_subjects, behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
POSTERIOR = 'posterior_mean'
fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects()

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)
    eids = behavioral_criterion(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) < 2:
        continue
    if len(eids) > 10:
        eids = eids[:10]

    # Get trial data
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        eids, stimulated='block', one=one)
    if len(session_uuids) == 0:
        continue

    # Fit model with one learning rate to get prior
    model = prev_action_1lr('./model_fit_results/', session_uuids, '%s_uncertainty_1lr' % nickname,
                            actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Get uncertainty cut-offs
    priors_array = np.concatenate(priors)[np.concatenate(prob_left) != 0]
    uncertain = ((priors > np.percentile(priors_array, (1/3)*100))
                 & (priors < np.percentile(priors_array, (2/3)*100)))

    # Get 4 combination vector
    opto_uncertain = np.zeros(stimulated.shape)
    opto_uncertain[(stimulated == 0) & (uncertain == 0)] = 0
    opto_uncertain[(stimulated == 0) & (uncertain == 1)] = 1
    opto_uncertain[(stimulated == 1) & (uncertain == 0)] = 2
    opto_uncertain[(stimulated == 1) & (uncertain == 1)] = 3
    opto_uncertain = torch.tensor(opto_uncertain.astype(int))

    # Fit model with four learning rates: low/high uncertainty and opto/no opto
    model = prev_action_4lr('./model_fit_results/', session_uuids, '%s_uncertainty_4lr' % nickname,
                              actions, stimuli, stim_side, opto_uncertain)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={
        'tau': [1/param_prevaction[0], 1/param_prevaction[1], 1/param_prevaction[2], 1/param_prevaction[3]],
        'uncertainty': ['low', 'high', 'low', 'high'], 'opto_stim': [0, 0, 1, 1],
        'sert-cre': subjects.loc[i, 'sert-cre'], 'subject': nickname}))

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
sns.lineplot(x='opto_stim', y='tau', hue='uncertainty', style='subject', estimator=None,
             data=results_df[results_df['sert-cre'] == 1], dashes=False,
             legend='brief', ax=ax1, palette=[colors['enhanced'], colors['suppressed']])
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='SERT',
        xticks=[0, 1], xticklabels=['No stim', 'Stim'])

sns.lineplot(x='opto_stim', y='tau', hue='uncertainty', style='subject', estimator=None,
             data=results_df[results_df['sert-cre'] == 0], dashes=False,
             legend=None, ax=ax2, palette=[colors['enhanced'], colors['suppressed']])
ax2.set(xlabel='', ylabel='Length of integration window (tau)', title='WT')

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'Low', 'High']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 5}, loc='center left', bbox_to_anchor=(1, .5))

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_opto_uncertainty.jpg'), dpi=600)

