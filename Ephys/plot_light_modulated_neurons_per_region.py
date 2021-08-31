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
ARTIFACT_ROC = 0.9

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Drop root
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'root' in j])

# Exclude artifact neurons
light_neurons = light_neurons[light_neurons['roc_auc'] < ARTIFACT_ROC]

# Calculate summary statistics
summary_df = light_neurons.groupby(['region', 'subject']).sum()
summary_df['n_neurons'] = light_neurons.groupby(['region', 'subject']).size()
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['full_region'] = get_full_region_name(summary_df['region'])
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = -summary_df['perc_supp']
ordered_regions = summary_df.groupby('full_region').max().sort_values(
                                'ratio', ascending=False).reset_index()

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=dpi)
sns.stripplot(x='perc_enh', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
plt.hlines(y=range(len(ordered_regions.index)), xmin=0, xmax=ordered_regions['perc_enh'],
           color='green')
plt.plot(ordered_regions['perc_enh'], range(len(ordered_regions.index)), 'o', color='green')
sns.stripplot(x='perc_supp', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
plt.hlines(y=range(len(ordered_regions.index)), xmin=ordered_regions['perc_supp'], xmax=0,
           color='red')
plt.plot(ordered_regions['perc_supp'], range(len(ordered_regions.index)), 'o', color='red')
ax1.plot([0, 0], ax1.get_ylim(), color=[0.5, 0.5, 0.5], ls='--')
ax1.set(ylabel='', xlabel='Percentage of significant neurons', xlim=[-51, 51],
        xticklabels=np.concatenate((np.arange(50, 0, -25), np.arange(0, 51, 25))))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulated_neurons_per_region'))
