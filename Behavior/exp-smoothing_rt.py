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
from serotonin_functions import paths, criteria_opto_eids, load_exp_smoothing_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
REMOVE_OLD_FIT = True
RT_CUTOFF = 0.3
POSTERIOR = 'posterior_mean'
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior-rt')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
accuracy_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range_blocks'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    elif subjects.loc[i, 'date_range_blocks'] == 'none':
        continue
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range_blocks'][:10],
                                      subjects.loc[i, 'date_range_blocks'][11:]])
    #eids = criteria_opto_eids(eids, max_lapse=0.5, max_bias=0.5, min_trials=200, one=one)
    if len(eids) == 0:
        continue

    # Get trial data
    actions, stimuli, stim_side, prob_left, stimulated, session_uuids = load_exp_smoothing_trials(
        eids, reaction_time=RT_CUTOFF, one=one)

    if len(session_uuids) == 0:
        continue

    # Fit models
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_rt' % nickname,
                            actions, stimuli, stim_side, torch.tensor(stimulated))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type=POSTERIOR)
    output_prevaction = model.compute_signal(signal=['prior', 'score'], act=actions, stim=stimuli, side=stim_side)
    priors_prevaction = output_prevaction['prior']
    results_df = results_df.append(pd.DataFrame(data={'tau_pa': [1/param_prevaction[0], 1/param_prevaction[1]],
                                                      'opto_stim': ['fast', 'slow'],
                                                      'sert-cre': subjects.loc[i, 'sert-cre'],
                                                      'subject': nickname}))

# %% Plot
figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', style='subject', estimator=None,
             data=results_df, dashes=False, markers=['o']*int(results_df.shape[0]/2),
             legend=False, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)', title=f'RT cutoff: {RT_CUTOFF} s')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'model_prevaction_reaction_time_{RT_CUTOFF}.png'))
