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
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS = 20
MIN_PERC = 0

# Paths
fig_path, save_path = paths()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Add enhanced and suppressed
light_neurons['enhanced_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] > 0)
light_neurons['suppressed_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] < 0)

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Calculate summary statistics
summary_df = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['perc_enh'] =  (summary_df['enhanced_late'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed_late'] / summary_df['n_neurons']) * 100
summary_df['perc_both'] =  summary_df['perc_supp'] + summary_df['perc_enh']
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = -summary_df['perc_supp']
# Exclude regions without modulation
summary_df = summary_df[(summary_df['perc_enh'] >= MIN_PERC) | (summary_df['perc_supp'] <= -MIN_PERC)]
# Get ordered regions
ordered_regions = summary_df.sort_values('perc_both', ascending=False).reset_index()

summary_no_df = light_neurons[light_neurons['expression'] == 0].groupby(['full_region']).sum()
summary_no_df['n_neurons'] = light_neurons[light_neurons['expression'] == 0].groupby(['full_region']).size()
summary_no_df['modulation_index'] = light_neurons[light_neurons['expression'] == 0].groupby(['full_region']).mean()['mod_index_late']
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_mod'] =  (summary_no_df['modulated'] / summary_no_df['n_neurons']) * 100
summary_no_df['perc_enh'] =  (summary_no_df['enhanced_late'] / summary_no_df['n_neurons']) * 100
summary_no_df['perc_supp'] =  (summary_no_df['suppressed_late'] / summary_no_df['n_neurons']) * 100
summary_no_df = summary_no_df[summary_no_df['n_neurons'] >= MIN_NEURONS]
summary_no_df['ratio'] = summary_no_df['perc_enh'] - summary_no_df['perc_supp']
summary_no_df['perc_supp'] = -summary_no_df['perc_supp']
ordered_regions_no = summary_no_df.sort_values('perc_mod', ascending=False).reset_index()


# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi)
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.stripplot(x='perc_enh', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
ax1.hlines(y=range(len(ordered_regions.index)), xmin=0, xmax=ordered_regions['perc_enh'],
           color=colors['enhanced'])
sns.stripplot(x='perc_supp', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)
ax1.hlines(y=range(len(ordered_regions.index)), xmin=ordered_regions['perc_supp'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp'], range(len(ordered_regions.index)), 'o', color=colors['suppressed'])
ax1.plot(ordered_regions['perc_enh'], range(len(ordered_regions.index)), 'o', color=colors['enhanced'])
ax1.set(ylabel='', xlabel='5-HT modulated neurons (%)', xlim=[-60, 30],
        xticklabels=np.concatenate((np.arange(60, 0, -20), np.arange(0, 41, 20))))
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))
ax1.margins(x=0)

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
ax2.set(ylabel='', xlabel='5-HT modulated neurons (%)', xlim=[-60, 30],
        xticklabels=np.concatenate((np.arange(60, 0, -20), np.arange(0, 41, 20))))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.png'))

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
sns.boxplot(x='mod_index_late', y='full_region', data=light_neurons[light_neurons['expression'] == 1],
            ax=ax1, fliersize=0, order=ordered_regions['full_region'], color='lightgrey')
ax1.plot([0, 0], [0, summary_df.shape[0]], color='r', ls='--')
ax1.set(ylabel='', xlim=[-0.7, 0.7], xticks=np.arange(-0.6, 0.61, 0.2), xlabel='Modulation index')

sns.boxplot(x='mod_index_late', y='full_region', data=light_neurons[light_neurons['expression'] == 0],
            ax=ax2, fliersize=0, order=ordered_regions['full_region'], color='lightgrey')
ax2.plot([0, 0], [0, summary_df.shape[0]], color='r', ls='--')
ax2.set(ylabel='', xlim=[-0.7, 0.7], xticks=np.arange(-0.6, 0.61, 0.2), xlabel='Modulation index')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_index_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulation_index_per_region.png'), dpi=300)

# %%
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
sns.barplot(x='full_region', y='perc_mod', data=summary_df.sort_values('perc_mod', ascending=False),
            color='orange', ax=ax1)
ax1.set(ylabel='5-HT modulated neurons (%)', xlabel='', ylim=[0, 60], yticks=np.arange(0, 61, 10))
ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
plt.xticks(rotation=90)
ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_region.png'), dpi=300)

