# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
from functions_pharmacology import paths, fit_probabilistic_choice_model

# Settings
FIG_PATH = paths()[1]
PLOT = True
PREVIOUS_TRIALS = 6

# Load in session dates
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)

# Load data
side_weights = pd.DataFrame()
other_weights = pd.DataFrame()
for i, nickname in enumerate(sessions['Nickname']):
    for j, condition in enumerate(['Pre-vehicle', 'Drug']):

        # Fit model
        fit_result = fit_probabilistic_choice_model(nickname, sessions.loc[i, condition],
                                                    previous_trials=PREVIOUS_TRIALS)
        success = []
        failure = []
        for t in range(PREVIOUS_TRIALS):
            success.append(fit_result['rchoice-%s' % str(t + 1)])
            failure.append(fit_result['uchoice-%s' % str(t + 1)])
        side_weights = side_weights.append(pd.DataFrame(data={
                                                'trial': np.arange(1, PREVIOUS_TRIALS+1),
                                                'subject': nickname,
                                                'success': success,
                                                'failure': failure,
                                                'week': sessions.loc[i, 'Week'],
                                                'condition': condition}), ignore_index=True)
        fit_results = fit_result[['1', '0.25', '0.125', '0.0625', 'const', 'block']]
        fit_results['subject'] = nickname
        fit_results['week'] = sessions.loc[i, 'Week']
        fit_results['condition'] = condition
        other_weights = other_weights.append(fit_results, ignore_index=True)


# %% Plot results
n_subjects = side_weights['subject'].unique().shape[0]
f, ax = plt.subplots(2, n_subjects + 1, figsize=(18, 8))
sns.set(context='paper', font_scale=1.5, style='ticks')

for i, subject in enumerate(side_weights['subject'].unique()):
    sns.lineplot(x='trial', y='success', data=side_weights[side_weights['subject'] == subject],
                 hue='condition', ci=68, palette='Dark2', ax=ax[0, i])
    ax[0, i].set(ylim=[-0.1, 1], ylabel='Weight of success term', title=subject,
                 xlabel='Trials in the past')

sns.lineplot(x='trial', y='success', data=side_weights, hue='condition', ci=68, palette='Dark2',
             ax=ax[0, -1])
ax[0, -1].set(ylim=[-0.1, 1], ylabel='Weight of success term',
              title='All mice', xlabel='Trials in the past')

for i, subject in enumerate(side_weights['subject'].unique()):
    sns.lineplot(x='trial', y='failure', data=side_weights[side_weights['subject'] == subject],
                 hue='condition', ci=68, palette='Dark2', ax=ax[1, i])
    ax[1, i].set(ylim=[-0.3, 0.6], ylabel='Weight of failure term',
                 title=subject, xlabel='Trials in the past')

sns.lineplot(x='trial', y='failure', data=side_weights, hue='condition', ci=68, palette='Dark2',
             ax=ax[1, -1])
ax[1, -1].set(ylim=[-0.3, 0.6], ylabel='Weight of failure term',
              title='All mice', xlabel='Trials in the past')

plt.tight_layout()
sns.despine(trim=True)
if PLOT:
    plt.savefig(join(FIG_PATH, 'probabilistic_choice_model_side_weights'))

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
colors = sns.color_palette(n_colors=3)

for i, subject in enumerate(other_weights['subject'].unique()):
    ax1.plot([other_weights.loc[(other_weights['condition'] == 'Pre-vehicle')
                                & (other_weights['subject'] == subject), '1'],
              other_weights.loc[(other_weights['condition'] == 'Drug')
                                & (other_weights['subject'] == subject), '1']],
             'o-', lw=2, color=colors[i], label=subject)
ax1.set(ylabel='Weight', xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='100% contrast')

for i, subject in enumerate(other_weights['subject'].unique()):
    ax2.plot([other_weights.loc[(other_weights['condition'] == 'Pre-vehicle')
                                & (other_weights['subject'] == subject), '0.25'],
              other_weights.loc[(other_weights['condition'] == 'Drug')
                                & (other_weights['subject'] == subject), '0.25']],
             'o-', lw=2, color=colors[i], label=subject)
ax2.set(xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='25% contrast')

for i, subject in enumerate(other_weights['subject'].unique()):
    ax3.plot([other_weights.loc[(other_weights['condition'] == 'Pre-vehicle')
                                & (other_weights['subject'] == subject), '0.125'],
              other_weights.loc[(other_weights['condition'] == 'Drug')
                                & (other_weights['subject'] == subject), '0.125']],
             'o-', lw=2, color=colors[i], label=subject)
ax3.set(xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='12.5% contrast')

for i, subject in enumerate(other_weights['subject'].unique()):
    ax4.plot([other_weights.loc[(other_weights['condition'] == 'Pre-vehicle')
                                & (other_weights['subject'] == subject), '0.0625'],
              other_weights.loc[(other_weights['condition'] == 'Drug')
                                & (other_weights['subject'] == subject), '0.0625']],
             'o-', lw=2, color=colors[i], label=subject)
ax4.set(xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='6.25% contrast')

for i, subject in enumerate(other_weights['subject'].unique()):
    ax5.plot([other_weights.loc[(other_weights['condition'] == 'Pre-vehicle')
                                & (other_weights['subject'] == subject), 'block'],
              other_weights.loc[(other_weights['condition'] == 'Drug')
                                & (other_weights['subject'] == subject), 'block']],
             'o-', lw=2, color=colors[i], label=subject)
ax5.set(xticks=[0, 1], xticklabels=['Pre-vehicle', 'Drug'], title='Stimulus prior')

plt.tight_layout()
sns.despine(trim=True)
if PLOT:
    plt.savefig(join(FIG_PATH, 'probabilistic_choice_model_other_weights'))

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
weight_names = ['1', '0.25', '0.125', '0.0625', 'block']
diff_weights = pd.DataFrame(columns=weight_names, data=(
        other_weights.loc[other_weights['condition'] == 'Drug', weight_names].values
        - other_weights.loc[other_weights['condition'] == 'Pre-vehicle', weight_names].values))
ax1.plot(diff_weights.transpose().values, 'ok')
ax1.plot([0, 5], [0, 0], '--', color=[0.6, 0.6, 0.6])
ax1.set(xticks=np.arange(5), xticklabels=['100%', '25%', '12.5%', '6.25%', 'Stim. prior'],
        ylabel='Drug induced increase in weight')
plt.xticks(rotation=45)

plt.tight_layout()
sns.despine(trim=True)
if PLOT:
    plt.savefig(join(FIG_PATH, 'probabilistic_choice_model_other_weights_diff'))
