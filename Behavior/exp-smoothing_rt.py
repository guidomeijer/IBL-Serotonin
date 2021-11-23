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
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
from serotonin_functions import (paths, load_exp_smoothing_trials, figure_style, load_subjects,
                                 query_opto_sessions, behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = True
RT_CUTOFF = 0.4
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Behavior', 'Models')

subjects = load_subjects(behavior=True)

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
    actions, stimuli, stim_side, prob_left, rt_low_high, session_uuids = load_exp_smoothing_trials(
        eids, stimulated='rt', rt_cutoff=RT_CUTOFF, one=one)

    if len(session_uuids) == 0:
        continue

    # Fit models
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_rt' % nickname,
                            actions, stimuli, stim_side, torch.tensor(rt_low_high))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    output_prevaction = model.compute_signal(signal=['prior', 'score'], act=actions, stim=stimuli, side=stim_side)
    priors_prevaction = output_prevaction['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'opto_stim': ['fast', 'slow'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))
results_df = results_df.reset_index(drop=True)
# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=None, ax=ax1, palette=[colors['wt'], colors['sert']])
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)', title=f'RT cutoff: {RT_CUTOFF} s')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'model_prevaction_reaction_time_{RT_CUTOFF}.png'))
