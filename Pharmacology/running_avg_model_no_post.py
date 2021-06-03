# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:40:23 2019

@author: guido
"""

from os.path import join
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from functions_pharmacology import paths

# Settings
DATA_PATH, FIG_PATH, _ = paths()

# Load in data
data = loadmat(join(DATA_PATH, 'guido_analysis_18mar2020.mat'))
X = data['X'][0]
parameters = data['pnames'][0]

results = pd.DataFrame(columns=['subject', 'condition', 'week', 'window_length', 'window_counts'])
for i in range(len(X)):
    for j in range(X[i].shape[1]):
        win_length = np.median(X[i][0][j][:, parameters[i][0] == 'runlength-tau'])
        win_counts = np.median(X[i][0][j][:, parameters[i][0] == 'beta-hyp'], axis=0)[1]
        results.loc[results.shape[0]+1] = ([i] + [np.mod(j, 3)]
                                           + [np.floor(j/3)+1] + [win_length] + [win_counts])

# Get bias normalized to pre-vehicle
results.loc[results['condition'] == 0, 'window_length_rel'] = (
                        results.loc[results['condition'] == 0, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 1, 'window_length_rel'] = (
                        results.loc[results['condition'] == 1, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 0, 'window_counts_rel'] = (
                        results.loc[results['condition'] == 0, 'window_counts'].values
                        / results.loc[results['condition'] == 0, 'window_counts'].values)
results.loc[results['condition'] == 1, 'window_counts_rel'] = (
                        results.loc[results['condition'] == 1, 'window_counts'].values
                        / results.loc[results['condition'] == 0, 'window_counts'].values)

# Get average over weeks per mouse
pre = []
drug = []
for i, subject in enumerate(results['subject'].unique()):
    pre.append(results.loc[(results['subject'] == subject)
                           & (results['condition'] == 0), 'window_length'].mean())
    drug.append(results.loc[(results['subject'] == subject)
                            & (results['condition'] == 1), 'window_length'].mean())

# %% Plot results
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
sns.set(context='paper', font_scale=1.5, style='ticks')
colors = sns.color_palette('colorblind', results['subject'].unique().shape[0])

for i, subject in enumerate(results['subject'].unique()):
    ax1.plot([results.loc[(results['condition'] == 0) & (results['subject'] == subject),
                          'window_length'],
              results.loc[(results['condition'] == 1) & (results['subject'] == subject),
                          'window_length']], 'o-', lw=2, color=colors[i], label=subject)
ax1.set(ylabel='Length of integration window (\u03C4 trials)', xticks=[0, 1],
        xticklabels=['Pre-vehicle', '5HT2a block'], title='All weeks')
# ax1.legend()

ax2.plot([pre, drug], 'o-', lw=2)
ax2.set(ylabel='Length of integration window (\u03C4 trials)', xticks=[0, 1],
        xticklabels=['Pre-vehicle', '5HT2a block'], title='Average over weeks')

ax3.plot([results.loc[(results['condition'] == 0) & (results['week'] == 1), 'window_length'],
          results.loc[(results['condition'] == 1) & (results['week'] == 1), 'window_length']],
         'o-', lw=2)
ax3.set(ylabel='Length of integration window (\u03C4 trials)', xticks=[0, 1],
        xticklabels=['Pre-vehicle', '5HT2a block'], title='Only first week')

sns.despine(trim=True, offset=10)
plt.tight_layout(pad=4)
plt.savefig(join(FIG_PATH, '5HT2a_block_integration_length'))

