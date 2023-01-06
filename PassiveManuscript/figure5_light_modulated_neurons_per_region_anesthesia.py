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
MIN_NEURONS_POOLED = 5
MIN_NEURONS_PER_MOUSE = 1
MIN_MOD_NEURONS = 1
MIN_REC = 1

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'])

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Get modulated neurons
mod_neurons = light_neurons[(light_neurons['sert-cre'] == 1) & (light_neurons['modulated'] == 1)]
mod_neurons = mod_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)

# Add enhanced and suppressed
light_neurons['enhanced_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] > 0)
light_neurons['suppressed_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] < 0)
light_neurons['enhanced_early'] = light_neurons['modulated'] & (light_neurons['mod_index_early'] > 0)
light_neurons['suppressed_early'] = light_neurons['modulated'] & (light_neurons['mod_index_early'] < 0)

# Calculate summary statistics
summary_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_enh_late'] =  (summary_df['enhanced_late'] / summary_df['n_neurons']) * 100
summary_df['perc_supp_late'] =  (summary_df['suppressed_late'] / summary_df['n_neurons']) * 100
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['modulated'] >= MIN_MOD_NEURONS]
summary_df['perc_supp_late'] = -summary_df['perc_supp_late']

summary_no_df = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).sum()
summary_no_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 0].groupby(['full_region']).size()
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_mod'] =  (summary_no_df['modulated'] / summary_no_df['n_neurons']) * 100
summary_no_df = summary_no_df[summary_no_df['modulated'] >= MIN_MOD_NEURONS]
summary_no_df = pd.concat((summary_no_df, pd.DataFrame(data={
    'full_region': summary_df.loc[~summary_df['full_region'].isin(summary_no_df['full_region']), 'full_region'],
    'perc_mod': np.zeros(np.sum(~summary_df['full_region'].isin(summary_no_df['full_region'])))})))

# Get ordered regions
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Summary statistics per mouse
per_mouse_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region', 'subject']).sum()
per_mouse_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)
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
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 82], xticks=np.arange(0, 81, 20))
ax1.legend(frameon=False, bbox_to_anchor=(0.5, 0.3))
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region.pdf'))


# %%
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)

sns.stripplot(x='perc_enh_late', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)  # this actually doesn't plot anything
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5])

ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=0, xmax=ordered_regions['perc_enh_late'],
           color=colors['enhanced'])
ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=ordered_regions['perc_supp_late'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp_late'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['suppressed'])
ax1.plot(ordered_regions['perc_enh_late'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['enhanced'])
ax1.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-60, 40],
        xticklabels=np.concatenate((np.arange(60, 0, -20), np.arange(0, 41, 20))))
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))
ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_region.pdf'))

# %%
colors, dpi = figure_style()

PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = mod_neurons.groupby('full_region').mean()['mod_index_late'].sort_values(ascending=False).reset_index()['full_region']

f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
sns.boxplot(x='mod_index_late', y='full_region', ax=ax1, data=mod_neurons, showmeans=True,
            order=ORDER, meanprops={"marker": "|", "markeredgecolor": "black", "markersize": "7"},
            fliersize=0, **PROPS)
sns.stripplot(x='mod_index_late', y='full_region', ax=ax1, data=mod_neurons, order=ORDER,
              size=2, palette=colors)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color=colors['grey'])
ax1.set(ylabel='', xlabel='Modulation index', xlim=[-1.05, 1.05], xticklabels=[-1, -0.5, 0, 0.5, 1])
#ax1.spines['bottom'].set_position(('data', np.floor(ax1.get_ylim()[0]) - 0.4))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_neuron_per_region.pdf'))