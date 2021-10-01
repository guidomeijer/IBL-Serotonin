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
MIN_NEURONS = 5
ARTIFACT_ROC = 0.5
MIN_PERC = 5

# Paths
_, fig_path, save_path = paths()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'void' in j])

# Add expression
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(light_neurons['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Exclude artifact neurons
light_neurons = light_neurons[light_neurons['roc_auc'] < ARTIFACT_ROC]

# Calculate summary statistics
summary_df = light_neurons[light_neurons['expression'] == 1].groupby(['region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['expression'] == 1].groupby(['region']).size()
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['full_region'] = get_full_region_name(summary_df['region'])
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = -summary_df['perc_supp']
# Exclude regions without modulation
summary_df = summary_df[(summary_df['perc_enh'] >= MIN_PERC) | (summary_df['perc_supp'] <= -MIN_PERC)]
# Get ordered regions
ordered_regions = summary_df.groupby('full_region').max().sort_values(
                                'ratio', ascending=False).reset_index()

summary_no_df = light_neurons[light_neurons['expression'] == 0].groupby(['region']).sum()
summary_no_df['n_neurons'] = light_neurons[light_neurons['expression'] == 0].groupby(['region']).size()
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_enh'] =  (summary_no_df['enhanced'] / summary_no_df['n_neurons']) * 100
summary_no_df['perc_supp'] =  (summary_no_df['suppressed'] / summary_no_df['n_neurons']) * 100
summary_no_df = summary_no_df[summary_no_df['n_neurons'] >= MIN_NEURONS]
summary_no_df['full_region'] = get_full_region_name(summary_no_df['region'])
summary_no_df['ratio'] = summary_no_df['perc_enh'] - summary_no_df['perc_supp']
summary_no_df['perc_supp'] = -summary_no_df['perc_supp']
ordered_regions_no = summary_no_df.groupby('full_region').max().sort_values(
                                'ratio', ascending=False).reset_index()


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.stripplot(x='perc_enh', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
ax1.hlines(y=range(len(ordered_regions.index)), xmin=0, xmax=ordered_regions['perc_enh'],
           color=colors['enhanced'])
ax1.plot(ordered_regions['perc_enh'], range(len(ordered_regions.index)), 'o', color=colors['enhanced'])
sns.stripplot(x='perc_supp', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
ax1.hlines(y=range(len(ordered_regions.index)), xmin=ordered_regions['perc_supp'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp'], range(len(ordered_regions.index)), 'o', color=colors['suppressed'])
ax1.set(ylabel='', xlabel='Percentage of serotonin modulated neurons', xlim=[-31, 31],
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
ax2.set(ylabel='', xlabel='Percentage of serotonin modulated neurons', xlim=[-51, 51],
        xticklabels=np.concatenate((np.arange(50, 0, -25), np.arange(0, 51, 25))))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region'))
