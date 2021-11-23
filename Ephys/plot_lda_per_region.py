#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, get_full_region_name

# Settings
MIN_NEURONS = 10
MIN_PERC = 3

# Paths
_, fig_path, save_path = paths()

# Load in results
lda_project = pd.read_csv(join(save_path, 'lda_opto_per_region.csv'))

# Drop root and void
lda_project = lda_project.reset_index(drop=True)
lda_project = lda_project.drop(index=[i for i, j in enumerate(lda_project['region']) if 'root' in j])
lda_project = lda_project.reset_index(drop=True)
lda_project = lda_project.drop(index=[i for i, j in enumerate(lda_project['region']) if 'void' in j])

# Get full region names
lda_project['full_region'] = get_full_region_name(lda_project['region'])

# Add expression
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(lda_project['subject'])):
    lda_project.loc[lda_project['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get ordered regions


# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
ordered_regions = lda_project[lda_project['sert-cre'] == 1].groupby('full_region').max().sort_values(
                                    'lda_dist', ascending=False).reset_index()
ax1.plot([0, 0], [0, lda_project.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.stripplot(x='lda_dist', y='full_region', data=lda_project[lda_project['sert-cre'] == 1], order=ordered_regions['full_region'],
              color='k', alpha=1, ax=ax1, size=4)
sns.barplot(x='lda_dist', y='full_region', data=lda_project[lda_project['sert-cre'] == 1], order=ordered_regions['full_region'],
            ax=ax1, ci=None, color=colors['general'])
ax1.set(ylabel='', xlabel='LDA projection')

ordered_regions = lda_project[lda_project['sert-cre'] == 0].groupby('full_region').max().sort_values(
                                    'lda_dist', ascending=False).reset_index()
ax2.plot([0, 0], [0, lda_project.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.stripplot(x='lda_dist', y='full_region', data=lda_project[lda_project['sert-cre'] == 0], order=ordered_regions['full_region'],
              color='k', alpha=1, ax=ax2, size=4)
sns.barplot(x='lda_dist', y='full_region', data=lda_project[lda_project['sert-cre'] == 0], order=ordered_regions['full_region'],
            ax=ax2, ci=None, color=colors['general'])
ax2.set(ylabel='', xlabel='LDA projection')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'LDA', 'lda_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'LDA', 'lda_per_region.png'), dpi=300)

