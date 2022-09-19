#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from serotonin_functions import (load_trials, plot_psychometric, paths, behavioral_criterion,
                                 fit_psychfunc, figure_style, query_opto_sessions, get_bias,
                                 load_subjects)
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()

# Settings
K = 2
INCLUDE_EPHYS = True
PLOT_SINGLE_ANIMALS = True
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')
subjects = load_subjects()

# Get subjects for which GLM-HMM data is available
glmhmm_subjects = os.listdir(join(data_path, 'GLM-HMM', 'results', 'individual_fit/'))

switch_df = pd.DataFrame()
for i, subject in enumerate(glmhmm_subjects):
    print(f'Starting {subject} ({i+1} of {len(glmhmm_subjects)})')

    # Load in model
    results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', subject)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Load in session data
    inpt, y, session = load_data(join(data_dir, subject + '_processed.npz'))
    all_sessions = np.unique(session)
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(np.hstack((inpt, np.ones((len(inpt), 1)))),
                                                           y, mask, session)

    # Get posterior probability of states per trial
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)

    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(np.unique(session)):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except Exception as err:
            print(err)
            continue
        if np.where(session == eid)[0].shape[0] != these_trials.shape[0]:
            print(f'Session {eid} mismatch')
            continue
        these_trials['state'] = states_max_posterior[np.where(session == eid)[0]]
        trials = pd.concat((trials, these_trials), ignore_index=True)

    trials = trials[trials['state'] == 0]

    # Make array of after block switch trials
    trials['block_switch'] = np.zeros(trials.shape[0])
    trial_blocks = (trials['probabilityLeft'] == 0.2).astype(int)
    block_trans = np.append(np.array(np.where(np.diff(trial_blocks) != 0)) + 1, [trial_blocks.shape[0]])

    for t, trans in enumerate(block_trans[:-1]):
        r_choice = trials.loc[(trials.index.values < block_trans[t+1])
                              & (trials.index.values >= block_trans[t]), 'right_choice'].reset_index(drop=True)
        if trials.loc[trans, 'probabilityLeft'] == 0.8:
            to_prior_choice = np.logical_not(r_choice).astype(int)
        else:
            to_prior_choice = r_choice.copy()
        switch_df = pd.concat((switch_df, pd.DataFrame(data={
            'right_choice': r_choice, 'trial': r_choice.index.values,
            'opto': trials.loc[trans, 'laser_stimulation'], 'to_prior_choice': to_prior_choice,
            'switch_to': trials.loc[trans, 'probabilityLeft'], 'subject': subject,
            'sert-cre': subjects.loc[i, 'sert-cre']})), ignore_index=True)

    if PLOT_SINGLE_ANIMALS:
        colors, dpi = figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
        sns.lineplot(x='trial', y='to_prior_choice', data=switch_df[switch_df['subject'] == subject],
                     hue='opto', errorbar='se', hue_order=[1, 0],
                     palette=[colors['stim'], colors['no-stim']])
        ax1.set(xlim=[0, 20], ylabel='Frac. choices to biased side', xlabel='Trials since switch',
                title=f'{subject}, SERT: {subjects.loc[i, "sert-cre"]}')
        leg_handles, _ = ax1.get_legend_handles_labels()
        leg_labels = ['Opto', 'No opto']
        ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left', frameon=False)
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(figure_dir, f'block_switch_{subject}.jpg'), dpi=300)
        asd

# %%

per_animal_df = switch_df.groupby(['subject', 'trial', 'opto']).mean().reset_index()

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
sns.lineplot(x='trial', y='to_prior_choice', data=per_animal_df[per_animal_df['sert-cre'] == 1],
             hue='opto', errorbar='se', hue_order=[1, 0],
             palette=[colors['stim'], colors['no-stim']], ax=ax1)
ax1.set(xlim=[0, 20], ylabel='Frac. choices to biased side', xlabel='Trials since switch',
        title='SERT')
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Opto', 'No opto']
ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left', frameon=False)

sns.lineplot(x='trial', y='to_prior_choice', data=per_animal_df[per_animal_df['sert-cre'] == 0],
             hue='opto', errorbar='se', hue_order=[1, 0],
             palette=[colors['stim'], colors['no-stim']], ax=ax2)
ax2.set(xlim=[0, 20], ylabel='Frac. choices to biased side', xlabel='Trials since switch',
        title='WT')
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['Opto', 'No opto']
ax2.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left', frameon=False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, f'all_block_switch.jpg'), dpi=300)

