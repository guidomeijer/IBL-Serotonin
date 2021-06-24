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
from serotonin_functions import paths, query_bwm_sessions, load_exp_smoothing_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = False
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

    # Fit model with one learning rate to get prior
    model = exp_prev_action_1('./model_fit_results/brain_wide_map', session_uuids, '%s' % nickname,
                              actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']

    # Get uncertainty cut-offs
    if len(priors.shape) == 1:
        priors_array = priors
    else:
        priors_array = np.concatenate(priors)[np.concatenate(prob_left) != 0]
    uncertain = (priors > np.percentile(priors_array, (1/3)*100)) & (priors < np.percentile(priors_array, (2/3)*100))

    # Fit model with two learning rates: one for high and one for low uncertainty
    uncertain = (priors < 0.75) & (priors > 0.25)
    if len(uncertain.shape) == 1:
        uncertain = [uncertain]
    model = exp_prev_action_2('./model_fit_results/brain_wide_map', session_uuids, '%s' % nickname,
                              actions, stimuli, stim_side, torch.tensor(uncertain))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    priors = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side)['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'uncertainty': ['low', 'high'],
                                                      'subject': nickname}))
    results_df.to_csv(join(save_path, 'brain_wide_map_exp_smoothing'))
results_df.to_csv(join(save_path, 'brain_wide_map_exp_smoothing'))

# %% Plot

# Discard outliers
plot_df = results_df[~results_df['subject'].isin(results_df.loc[results_df['tau'] > 50, 'subject'])]

colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
sns.lineplot(x='uncertainty', y='tau', style='subject', estimator=None, lw=2, ms=8,
             data=plot_df, dashes=False, markers=['o']*int(plot_df.shape[0]/2), legend=False, ax=ax1)
ax1.set(xlabel='Uncertainty', ylabel='Length of integration window (tau)')

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_bwm_uncertainty'))

