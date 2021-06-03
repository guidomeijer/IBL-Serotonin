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
from functions_pharmacology import paths

# Settings
DATA_PATH, FIG_PATH, _ = paths()

# Load in data
data = loadmat(join(DATA_PATH, 'guido_analysis_18mar2020.mat'))
X = data['X'][0]
parameters = data['pnames'][0]

results = pd.DataFrame(columns=['subject', 'condition', 'week', 'window_length', 'iqr'])
for i in range(len(X)):
    for j in range(X[i].shape[1]):
        max_prob = np.mean(X[i][0][j][:, parameters[i][0] == 'runlength-tau'])
        iqr = stats.iqr(X[i][0][j][:, parameters[i][0] == 'runlength-tau'])
        results.loc[results.shape[0]+1] = ([i] + [np.mod(j, 3)]
                                           + [np.floor(j/3)+1] + [max_prob] + [iqr])

# Get bias normalized to pre-vehicle
results.loc[results['condition'] == 0, 'window_length_rel'] = (
                        results.loc[results['condition'] == 0, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 1, 'window_length_rel'] = (
                        results.loc[results['condition'] == 1, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 2, 'window_length_rel'] = (
                        results.loc[results['condition'] == 2, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)

# %% Plot results
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
sns.set(context='paper', font_scale=1.5, style='ticks')

sns.lineplot(x='condition', y='window_length', units='subject', estimator=None, color='black',
             sort=False, data=results[(results['week'] == 1)], lw=2, ax=ax1)
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        ylabel='Length integration window (\u03C4 trials)', title='Week 1', ylim=[2, 8])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
# ax1.get_legend().remove()

sns.lineplot(x='condition', y='window_length', units='subject', estimator=None, hue='subject',
             sort=False, data=results[(results['week'] == 2)], ax=ax2)
ax2.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        ylabel='Length integration window (\u03C4 trials)', title='Week 2')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
ax2.get_legend().remove()

sns.lineplot(x='condition', y='window_length', units='subject', estimator=None, hue='subject',
             sort=False, data=results[(results['week'] == 3)], ax=ax3)
ax3.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        ylabel='Length integration window (\u03C4 trials)', title='Week 3')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)
ax3.get_legend().remove()

sns.despine(trim=True)
plt.tight_layout(pad=2)

plt.savefig(join(FIG_PATH, '5HT2a_block_integration_length_per_week'), dpi=300)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
sns.set(context='paper', font_scale=1.5, style='ticks')

h_plot = sns.lineplot(x='condition', y='window_length', hue='week', units='subject',
                      estimator=None, data=results, legend='brief', ax=ax1, lw=3,
                      palette=sns.cubehelix_palette(results['week'].unique().shape[0],
                                                    reverse=True))
leg = h_plot.legend_
for t in leg.texts[1:]:
    # truncate label text to 4 characters
    t.set_text(t.get_text()[:1])
"""
sns.lineplot(x='condition', y='window_length', hue='week', style='week', units='subject',
             estimator=None, data=results, legend=False, ax=ax1, lw=3,
             palette=sns.cubehelix_palette(results['week'].unique().shape[0], reverse=True),
             markers=['o']*results['week'].unique().shape[0], markersize=8, dashes=False)
"""
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Length of integration window (\u03C4 trials)')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.lineplot(x='condition', y='window_length_rel', hue='subject', style='week', data=results,
             legend=False, ax=ax2, lw=3)
ax2.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Length of integration window (\u03C4 trials)')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

# sns.boxplot(x='condition', y='window_length', data=results, ax=ax3)
sns.lineplot(x='condition', y='window_length', data=results,
             legend=False, ax=ax3, lw=3, ci=68)
ax3.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Length of integration window (\u03C4 trials)')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)

sns.lineplot(x='condition', y='window_length_rel', data=results,
             legend=False, ax=ax4, lw=3, ci=68)
ax4.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Length of integration window (\u03C4 trials)')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)

sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine(trim=True)
plt.tight_layout(pad=3)

plt.savefig(join(FIG_PATH, '5HT2a_block_integration_length'), dpi=300)
