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
MIN_NEURONS_PER_MOUSE = 5
MIN_MOD_NEURONS = 10
MIN_REC = 2

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure2')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))

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
per_mouse_df = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['full_region', 'subject']).sum(numeric_only=True)
per_mouse_df['mod_index_late'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(
    ['full_region', 'subject']).median(numeric_only=True)['mod_index_late']
per_mouse_df['n_neurons'] = light_neurons[light_neurons['sert-cre'] == 1].groupby(['full_region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)
per_mouse_df = per_mouse_df.reset_index()

# Add mouse number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    per_mouse_df.loc[per_mouse_df['subject'] == nickname, 'subject_nr'] = int(subjects.loc[subjects['subject'] == nickname].index[0])
per_mouse_df['subject_nr'] = per_mouse_df['subject_nr'].astype(int)

# Get ordered regions per mouse
ordered_regions_pm = per_mouse_df.groupby('full_region').mean(numeric_only=True).sort_values('perc_mod', ascending=False).reset_index()

# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=per_mouse_df, order=ordered_regions_pm['full_region'],
            color=[0.6, 0.6, 0.6], ax=ax1, errorbar=None)

sns.swarmplot(x='perc_mod', y='full_region', data=per_mouse_df, order=ordered_regions_pm['full_region'],
              palette='tab20', hue='subject_nr', ax=ax1, size=2)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 100], xticks=np.arange(0, 101, 20))

#handles, labels = ax1.get_legend_handles_labels()
#new_labels = [label[:-3] for label in labels]
#ax1.legend(handles, new_labels, title = 'hr')


ax1.legend(frameon=False, bbox_to_anchor=(1, 0.85), prop={'size': 5}, title='Mice')
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'perc_light_modulated_neurons_per_region.pdf'))



# %%
colors, dpi = figure_style()

PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = mod_neurons.groupby('full_region').mean()['mod_index_late'].sort_values(ascending=False).reset_index()['full_region']

f, ax1 = plt.subplots(1, 1, figsize=(2.9, 2.5), dpi=dpi)
sns.stripplot(x='mod_index_late', y='full_region', ax=ax1, data=mod_neurons, order=ORDER,
              size=2, color='grey', zorder=1)
sns.boxplot(x='mod_index_late', y='full_region', ax=ax1, data=mod_neurons, showmeans=True,
            order=ORDER, meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=2, **PROPS)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', zorder=0)
ax1.set(ylabel='', xlabel='Modulation index', xlim=[-1.05, 1.05], xticklabels=[-1, -0.5, 0, 0.5, 1])
#ax1.spines['bottom'].set_position(('data', np.floor(ax1.get_ylim()[0]) - 0.4))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_neuron_per_region.pdf'))

# %%

f, ax1 = plt.subplots(1, 1, figsize=(2.9, 2.5), dpi=dpi)
sns.stripplot(x='mod_index_late', y='full_region', ax=ax1, data=per_mouse_df, order=ORDER,
              size=2, color='grey', zorder=1)
sns.boxplot(x='mod_index_late', y='full_region', ax=ax1, data=per_mouse_df, showmeans=True,
            order=ORDER, meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=2, **PROPS)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', zorder=0)
ax1.set(ylabel='', xlabel='Modulation index', xlim=[-0.3, 0.1])
#ax1.spines['bottom'].set_position(('data', np.floor(ax1.get_ylim()[0]) - 0.4))
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_mouse_per_region.pdf'))


