#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:03:38 2021

@author: guido
"""
from serotonin_functions import load_exp_smoothing_trials, figure_style, paths
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import pandas as pd
import numpy as np
import torch
from oneibl.one import ONE
from models.expSmoothing_prevAction_SE import expSmoothing_prevAction_SE as exp_prev_action
one = ONE()

REMOVE_OLD_FIT = True
TRIALS_AFTER_SWITCH = 20

_, fig_path, _ = paths()
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)


def fit_model(eids):
    # Load trials
    actions, stimuli, stim_side, prob_left, eids = load_exp_smoothing_trials(eids, one=one)

    # Make array of after block switch trials
    block_switch = np.zeros(prob_left.shape)
    for k in range(prob_left.shape[0]):
        ses_length = np.sum(prob_left[k] != 0)
        trial_blocks = (prob_left[k][:ses_length] == 0.2).astype(int)
        block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
        block_trans = np.append(block_trans, [trial_blocks.shape[0]])
        for s, ind in enumerate(block_trans):
            block_switch[k][ind:ind+TRIALS_AFTER_SWITCH] = 1

    # Fit model
    model = exp_prev_action('./model_fit_results/', eids, nickname, actions, stimuli, stim_side,
                            torch.tensor(block_switch))
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    params = model.get_parameters(parameter_type='posterior_mean')
    tau_early_block = 1/params[1]
    tau_late_block = 1/params[0]
    return tau_early_block, tau_late_block


results_df = pd.DataFrame()
for i, nickname in enumerate(sessions['Nickname'].unique()):
    # Pre-vehicle
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Pre-vehicle']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])
    tau_early_block, tau_late_block = fit_model(eids)
    results_df = results_df.append(pd.DataFrame(data={'tau': [tau_early_block, tau_late_block],
                                                      'After switch': ['Yes', 'No'],
                                                      'Subject': nickname,
                                                      'condition': 'Pre-vehicle'}))

    # Drug
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Drug']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])
    tau_early_block, tau_late_block = fit_model(eids)
    results_df = results_df.append(pd.DataFrame(data={'tau': [tau_early_block, tau_late_block],
                                                      'After switch': ['Yes', 'No'],
                                                      'Subject': nickname,
                                                      'condition': 'Drug'}))

    # Post-vehicle
    eids = []
    for s, date in enumerate(sessions.loc[sessions['Nickname'] == nickname, 'Post-vehicle']):
        eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
        eids.append(eid[0])
    tau_early_block, tau_late_block = fit_model(eids)
    results_df = results_df.append(pd.DataFrame(data={'tau': [tau_early_block, tau_late_block],
                                                      'After switch': ['Yes', 'No'],
                                                      'Subject': nickname,
                                                      'condition': 'Post-vehicle'}))
# %% Plot
colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
sns.lineplot(x='condition', y='tau', hue='After switch', style='Subject', estimator=None,
             data=results_df,
             dashes=False,
             legend='brief', lw=2, ms=8, ax=ax1)
#handles, labels = ax1.get_legend_handles_labels()
#labels = ['', 'WT', 'SERT']
#ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 20}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='Trials after block switch', ylabel='Length of integration window (tau)', ylim=[0, 20])

sns.despine(trim=True, offset=10)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_prevaction_block-switch_pharm'))
