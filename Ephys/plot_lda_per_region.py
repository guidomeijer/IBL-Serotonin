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
    lda_project.loc[lda_project['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    lda_project.loc[lda_project['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get ordered regions
ordered_regions = lda_project.groupby('full_region').max().sort_values(
                                    'lda_dist', ascending=False).reset_index()


# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
ax1.plot([0, 0], [0, lda_project.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.stripplot(x='lda_dist', y='full_region', data=lda_project, order=ordered_regions['full_region'],
              color='k', alpha=1, ax=ax1)
sns.barplot(x='lda_dist', y='full_region', data=lda_project, order=ordered_regions['full_region'],
            ax=ax1, )


ax1.plot(ordered_regions['perc_enh'], range(len(ordered_regions.index)), 'o', color=colors['enhanced'])
sns.stripplot(x='perc_supp', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
ax1.hlines(y=range(len(ordered_regions.index)), xmin=ordered_regions['perc_supp'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp'], range(len(ordered_regions.index)), 'o', color=colors['suppressed'])
ax1.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-31, 31],
        xticklabels=np.concatenate((np.arange(30, 0, -10), np.arange(0, 31, 10))))
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))

sns.stripplot(x='perc_enh', y='full_region', data=summary_no_df, order=ordered_regions_no['full_region'],
              color='k', alpha=0, ax=ax2)
ax2.hlines(y=range(len(ordered_regions_no.index)), xmin=0, xmax=ordered_regions_no['perc_enh'],
           color=colors['enhanced'])
ax2.plot(ordered_regions_no['perc_enh'], range(len(ordered_regions_no.index)), 'o', color=colors['enhanced'])
sns.stripplot(x='perc_supp', y='full_region', data=summary_no_df, order=ordered_regions_no['full_region'],
              color='k', alpha=0, ax=ax2)
ax2.hlines(y=range(len(ordered_regions_no.index)), xmin=ordered_regions_no['perc_supp'], xmax=0,
           color=colors['suppressed'])
ax2.plot(ordered_regions_no['perc_supp'], range(len(ordered_regions_no.index)), 'o', color=colors['suppressed'])
ax2.plot([0, 0], ax2.get_ylim(), color=[0.5, 0.5, 0.5], ls='--')
ax2.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-51, 51],
        xticklabels=np.concatenate((np.arange(50, 0, -25), np.arange(0, 51, 25))))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.png'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
sns.boxplot(x='roc_auc', y='full_region', data=lda_project, ax=ax1, fliersize=0,
            order=ordered_regions['full_region'], color='lightgrey')
ax1.plot([0, 0], [0, summary_df.shape[0]], color='r', ls='--')
#sns.displot(x='roc_auc', y='full_region', data=lda_project, ax=ax1,
#            order=ordered_regions['full_region'], palette='coolwarm_r')
ax1.set(ylabel='', xlim=[-0.6, 0.6], xticks=np.arange(-0.6, 0.61, 0.2), xlabel='Modulation index')

ax2.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.violinplot(x='modulation_index', y='full_region', data=summary_no_df, ax=ax2,
               order=ordered_regions['full_region'], palette='coolwarm_r')
ax2.set(ylabel='', xlim=[-0.15, 0.15], xticks=np.arange(-0.15, 0.16, 0.05), xlabel='Modulation index')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_index_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_index_per_region.png'))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
sns.kdeplot(data=lda_project, x='roc_auc', hue='region', ax=ax1,
            hue_order=ordered_regions['region'], palette='coolwarm_r')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_distribution_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_distribution_per_region.png'))
