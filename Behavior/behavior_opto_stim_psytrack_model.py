#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:31:56 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from psytrack.hyperOpt import hyperOpt
from serotonin_functions import paths, criteria_opto_eids, load_trials, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
PLOT_SINGLE = False
PREV_TRIALS = 0
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv(join('..', 'subjects.csv'))

results_df = pd.DataFrame()
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
    contrast_l, contrast_r, prob_l, correct, choice, opto_stim, day_length = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    for k, eid in enumerate(eids):
        trials = load_trials(eid, laser_stimulation=True, one=one)
        if trials is None:
            continue
        if 'laser_stimulation' not in trials.columns:
            continue
        contrast_l = np.append(contrast_l, trials.loc[trials['choice'] != 0, 'contrastLeft'])
        contrast_r = np.append(contrast_r, trials.loc[trials['choice'] != 0, 'contrastRight'])
        prob_l = np.append(prob_l, trials.loc[trials['choice'] != 0, 'probabilityLeft'])
        correct = np.append(correct, trials.loc[trials['choice'] != 0, 'correct'])
        choice = np.append(choice, trials.loc[trials['choice'] != 0, 'choice'])
        opto_stim = np.append(choice, trials.loc[trials['choice'] != 0, 'laser_stimulation'])
        day_length = np.append(day_length, trials[trials['choice'] != 0].shape[0])

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for j in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(j+j),
                                                        contrast_l_transform[j:-j])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(j+j),
                                                        contrast_r_transform[j:-j])))

    # Create input dict
    D = {'name': nickname, 'y': choice, 'correct': correct, 'dayLength': day_length,
         'inputs': {'s1': s1_trans, 's2': s2_trans}}

    # Model parameters
    weights = {'bias': 1, 's1': PREV_TRIALS+1, 's2': PREV_TRIALS+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4., 'sigma': [2**-4.]*K, 'sigDay': [2**-4.]*K}
    optList = ['sigInit', 'sigma', 'sigDay']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)
    results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
        'subject': nickname, 'sert-cre': subjects.loc[i, 'sert-cre'], 'volatility': hyp['sigma'][0]}))

    # Plot output
    if PLOT_SINGLE:
        figure_style()
        f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
        block_switch = np.where(np.abs(np.diff(prob_l)) > 0.1)[0]
        block_switch = np.concatenate(([0], block_switch+1, [np.size(prob_l)]), axis=0)
        for i, ind in enumerate(block_switch[:-1]):
            if prob_l[block_switch[i]] == 0.5:
                ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                         [-4, 4, 4, -4], color=[0.7, 0.7, 0.7])
            if prob_l[block_switch[i]] == 0.2:
                ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                         [-4, 4, 4, -4], color=[0.6, 0.6, 1])
            if prob_l[block_switch[i]] == 0.8:
                ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                         [-4, 4, 4, -4], color=[1, 0.6, 0.6])
        ax1.plot(wMode[0], color='k', lw=3)
        ax1.set(ylabel='Weight', xlabel='Trials', ylim=[-4, 4])
        sns.set(context='paper', font_scale=1.5, style='ticks')
        sns.despine(trim=True)
        plt.tight_layout(pad=2)
        plt.savefig(join(fig_path, f'{nickname}_model_psytrack'))

# %% Plot summary

colors = figure_style(return_colors=True)
f, ax1 = plt.subplots(1, 1, figsize=(4, 5), dpi=150)
sns.stripplot(x='sert-cre', y='volatility', data=results_df, s=8,
              palette=[colors['wt'], colors['sert']], ax=ax1)
ax1.set(ylabel='Volatility', xlabel='', ylim=[0, 1], xticks=[0, 1], xticklabels=['WT', 'Sert-Cre'])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_psytrack_volatility'))
