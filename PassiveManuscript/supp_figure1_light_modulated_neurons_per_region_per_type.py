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
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure3')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))

# Merge dataframes
neuron_type['neuron_id'] = neuron_type['cluster_id']
merged_df = pd.merge(light_neurons, neuron_type, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])

# Get full region names
merged_df['full_region'] = combine_regions(merged_df['region'], split_thalamus=False)
#light_neurons['full_region'] = light_neurons['region']

# Drop neurons that could not be defined as RS or FS
merged_df = merged_df[(merged_df['type'] == 'RS') | (merged_df['type'] == 'FS')]

# Drop root and void
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['full_region']) if 'root' in j])
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['full_region']) if 'void' in j])

# Add enhanced and suppressed
merged_df['modulated_FS'] = merged_df['modulated'] & (merged_df['type'] == 'FS')
merged_df['modulated_RS'] = merged_df['modulated'] & (merged_df['type'] == 'RS')
merged_df['enhanced_FS'] = merged_df['modulated'] & (merged_df['mod_index_late'] > 0) & (merged_df['type'] == 'FS')
merged_df['suppressed_FS'] = merged_df['modulated'] & (merged_df['mod_index_late'] < 0) & (merged_df['type'] == 'FS')
merged_df['enhanced_RS'] = merged_df['modulated'] & (merged_df['mod_index_late'] > 0) & (merged_df['type'] == 'RS')
merged_df['suppressed_RS'] = merged_df['modulated'] & (merged_df['mod_index_late'] < 0) & (merged_df['type'] == 'RS')

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
merged_df = merged_df[merged_df['sert-cre'] == 1]

# Calculate summary statistics
summary_df = merged_df.groupby(['full_region']).sum()
summary_df['n_neurons'] = merged_df.groupby(['full_region']).size()
summary_df['modulation_index_FS'] = merged_df[merged_df['type'] == 'FS'].groupby(['full_region']).mean()['mod_index_late']
summary_df['modulation_index_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_enh_FS'] =  (summary_df['enhanced_FS'] / summary_df['n_neurons']) * 100
summary_df['perc_supp_FS'] =  (summary_df['suppressed_FS'] / summary_df['n_neurons']) * 100
summary_df['perc_enh_RS'] =  (summary_df['enhanced_RS'] / summary_df['n_neurons']) * 100
summary_df['perc_supp_RS'] =  (summary_df['suppressed_RS'] / summary_df['n_neurons']) * 100
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['perc_mod_RS'] =  (summary_df['modulated_RS'] / summary_df['n_neurons']) * 100
summary_df['perc_mod_FS'] =  (summary_df['modulated_FS'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['perc_supp_FS'] = -summary_df['perc_supp_FS']
summary_df['perc_supp_RS'] = -summary_df['perc_supp_RS']

# Get ordered regions by percentage modulated
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Calculate percentage of FS and RS modulated neurons per region
summary_df['perc_RS_mod'] = (summary_df['perc_mod_RS'] /
                             (summary_df['perc_mod_RS'] + summary_df['perc_mod_FS'])) * 100
summary_df['perc_FS_mod'] = (summary_df['perc_mod_FS'] /
                             (summary_df['perc_mod_RS'] + summary_df['perc_mod_FS'])) * 100
summary_df['100perc'] = 100

# Get ordered regions by ratio FS/NS
ordered_regions_FS = summary_df.sort_values('perc_FS_mod', ascending=False).reset_index()

# %% Plot ratio FS/RS modulated neurons

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3.75, 2), dpi=dpi)
sns.barplot(x='100perc', y='full_region', data=summary_df, color=colors['RS'], ax=ax1,
            order=ordered_regions_FS['full_region'], label='Regular spiking')
sns.barplot(x='perc_FS_mod', y='full_region', data=summary_df, color=colors['FS'], ax=ax1,
            order=ordered_regions_FS['full_region'], label='Fast spiking\ninterneurons')
#summary_df[['perc_RS_mod', 'perc_FS_mod']].plot(kind='bar', stacked=True)
ax1.set(ylabel='', xlabel='Percentage of modulated neurons')
ax1.legend(frameon=False, bbox_to_anchor=(0.98, 1))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'ratio_mod_neurons.pdf'))

# %%  Plot percentage of enhanced and suppressed neurons per type and region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2.5), dpi=dpi)
DIST = 0.15
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5])
sns.stripplot(x='perc_enh_FS', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)  # this actually doesn't plot anything
ax1.hlines(y=np.arange(ordered_regions.shape[0])-DIST, xmin=0, xmax=ordered_regions['perc_enh_FS'],
           color=colors['FS'])
ax1.hlines(y=np.arange(ordered_regions.shape[0])-DIST, xmin=ordered_regions['perc_supp_FS'], xmax=0,
           color=colors['FS'])
ax1.plot(ordered_regions['perc_supp_FS'], np.arange(ordered_regions.shape[0])-DIST, 'o',
         color=colors['FS'])
ax1.plot(ordered_regions['perc_enh_FS'], np.arange(ordered_regions.shape[0])-DIST, 'o',
         color=colors['FS'])
ax1.hlines(y=np.arange(ordered_regions.shape[0])+DIST, xmin=0, xmax=ordered_regions['perc_enh_RS'],
           color=colors['RS'])
ax1.hlines(y=np.arange(ordered_regions.shape[0])+DIST, xmin=ordered_regions['perc_supp_RS'], xmax=0,
           color=colors['RS'])
ax1.plot(ordered_regions['perc_supp_RS'], np.arange(ordered_regions.shape[0])+DIST, 'o',
         color=colors['RS'])
ax1.plot(ordered_regions['perc_enh_RS'], np.arange(ordered_regions.shape[0])+DIST, 'o',
         color=colors['RS'])
ax1.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-40, 20], xticks=[-40, -20, 0, 20],
        xticklabels=[40, 20, 0, 20])
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))
ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_modulation_per_region.pdf'))