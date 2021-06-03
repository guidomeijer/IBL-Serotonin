#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:40:36 2020

@author: guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from functions_pharmacology import paths, fit_psytrack_multiple_days

# Settings
TRIAL_WIN = [-5, 20]

# Load data
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)
results = pd.DataFrame(columns=['Nickname', 'Condition', 'Bias'])
bias_results = pd.DataFrame()
for i, nickname in enumerate(sessions['Nickname'].unique()):
    print('Starting subject %s' % nickname)
    for j, condition in enumerate(['Pre-vehicle', 'Drug']):

        # Fit psytrack model over all weeks
        print('Condition: %s' % condition)
        wMode, prob_l, hyp = fit_psytrack_multiple_days(
                                sessions.loc[i, 'Nickname'],
                                sessions.loc[sessions['Nickname'] == nickname, condition].value)
        results.loc[results.shape[0]+1] = ([nickname] + [condition] + [hyp['sigma'][0]])

        # Get bias centered at block change
        left_blocks = np.where(np.diff(prob_l) > 0.2)[0]
        left_blocks = left_blocks[left_blocks < prob_l.shape[0] - TRIAL_WIN[1]]
        for l, ind in enumerate(left_blocks):
            bias_results = bias_results.append(pd.DataFrame(data={
                                    'bias': (wMode[0][ind+TRIAL_WIN[0]:ind+TRIAL_WIN[1]]
                                             - wMode[0][ind+TRIAL_WIN[0]:ind-1].mean()),
                                    'trials': np.append(np.arange(TRIAL_WIN[0], 0),
                                                        np.arange(1, TRIAL_WIN[1] + 1)),
                                    'switch': 'left', 'condition': condition,
                                    'nickname': nickname}))
        right_blocks = np.where(np.diff(prob_l) < -0.2)[0]
        right_blocks = right_blocks[right_blocks < prob_l.shape[0] - TRIAL_WIN[1]]
        for l, ind in enumerate(right_blocks):
            bias_results = bias_results.append(pd.DataFrame(data={
                                    'bias': (wMode[0][ind+TRIAL_WIN[0]:ind+TRIAL_WIN[1]]
                                             - wMode[0][ind+TRIAL_WIN[0]:ind-1].mean()),
                                    'trials': np.append(np.arange(TRIAL_WIN[0], 0),
                                                        np.arange(1, TRIAL_WIN[1] + 1)),
                                    'switch': 'right', 'condition': condition,
                                    'nickname': nickname}))

# %% Plot

colors = sns.color_palette('Dark2', n_colors=2)
f, ax = plt.subplots(1, bias_results['nickname'].unique().shape[0],
                     figsize=(6 * bias_results['nickname'].unique().shape[0], 6))
for i, subject in enumerate(bias_results['nickname'].unique()):
    sns.lineplot(x='trials', y='bias', data=bias_results[bias_results['nickname'] == subject],
                 hue='condition', style='switch', palette='Dark2', ci=68, legend=None, ax=ax[i])
    ax[i].set(title=subject, ylabel='Baseline subtracted bias', xlabel='Trials from block switch')

sns.despine(trim=True)
plt.tight_layout(pad=4)
plt.savefig(join(paths()[1], 'psytrack_altanserin_fit_over_weeks_per_subject'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.lineplot(x='Condition', y='Bias', data=results, hue='Nickname', units='Nickname',
             estimator=None, sort=False, lw=2, ax=ax1)
ax1.set(ylabel='Bias volatility parameter')

sns.lineplot(x='trials', y='bias', data=bias_results, hue='condition', style='switch',
             ci=68, palette='Dark2', legend=None, ax=ax2)
ax2.set(ylabel='Baseline subtracted bias', xlabel='Trials from block switch')

# sns.despine(trim=True)
plt.tight_layout(pad=4)
plt.savefig(join(paths()[1], 'psytrack_altanserin_fit_over_weeks'))
