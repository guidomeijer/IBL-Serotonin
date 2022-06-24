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
MIN_NEURONS_POOLED = 20
MIN_NEURONS_PER_MOUSE = 10

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'Task')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)
#light_neurons['full_region'] = light_neurons['region']

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Add enhanced and suppressed
light_neurons['enhanced'] = light_neurons['opto_modulated'] & (light_neurons['opto_mod_roc'] > 0)
light_neurons['suppressed'] = light_neurons['opto_modulated'] & (light_neurons['opto_mod_roc'] < 0)

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Calculate summary statistics
summary_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).mean()['opto_mod_roc']
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed'] / summary_df['n_neurons']) * 100
summary_df['perc_mod'] =  (summary_df['opto_modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS_POOLED]
summary_df['perc_supp'] = -summary_df['perc_supp']

summary_no_df = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).sum()
summary_no_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).size()
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_mod'] =  (summary_no_df['opto_modulated'] / summary_no_df['n_neurons']) * 100
summary_no_df = summary_no_df[summary_no_df['n_neurons'] >= MIN_NEURONS_POOLED]
summary_no_df = pd.concat((summary_no_df, pd.DataFrame(data={
    'full_region': summary_df.loc[~summary_df['full_region'].isin(summary_no_df['full_region']), 'full_region'],
    'perc_mod': np.zeros(np.sum(~summary_df['full_region'].isin(summary_no_df['full_region'])))})))

# Get ordered regions
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Summary statistics per mouse
per_mouse_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region', 'subject']).sum()
per_mouse_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['opto_modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.reset_index()

# Get ordered regions per mouse
ordered_regions_pm = per_mouse_df.groupby('full_region').mean().sort_values('perc_mod', ascending=False).reset_index()

# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=summary_df, order=ordered_regions['full_region'],
            color=colors['sert'], ax=ax1, label='SERT')
sns.barplot(x='perc_mod', y='full_region', data=summary_no_df, order=ordered_regions['full_region'],
            color=colors['wt'], ax=ax1, label='WT')
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 50], xticks=np.arange(0, 51, 10))
ax1.legend(frameon=False, bbox_to_anchor=(0.5, 0.3))
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region_pooled.pdf'))


# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=per_mouse_df, order=ordered_regions_pm['full_region'],
            color=colors['sert'], ax=ax1, ci=None)
sns.swarmplot(x='perc_mod', y='full_region', data=per_mouse_df, order=ordered_regions_pm['full_region'],
              color=colors['grey'], ax=ax1, size=2)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 102], xticks=np.arange(0, 101, 25))
ax1.legend(frameon=False, bbox_to_anchor=(0.5, 0.3))
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region.pdf'))


# %%
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)

sns.stripplot(x='perc_enh', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)  # this actually doesn't plot anything
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5])

ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=0, xmax=ordered_regions['perc_enh'],
           color=colors['enhanced'])
ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=ordered_regions['perc_supp'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['suppressed'])
ax1.plot(ordered_regions['perc_enh'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['enhanced'])
ax1.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-60, 40],
        xticklabels=np.concatenate((np.arange(60, 0, -20), np.arange(0, 41, 20))))
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))
ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_region.pdf'))