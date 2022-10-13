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
import seaborn.objects as so
from os.path import join
from scipy.stats import pearsonr
from serotonin_functions import paths, figure_style, combine_regions, load_subjects, high_level_regions

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
#merged_df['full_region'] = high_level_regions(merged_df['region'])
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

# Calculate stats per animal
per_animal_df = merged_df.groupby(['full_region', 'subject']).sum()
per_animal_df['n_neurons'] = merged_df.groupby(['full_region', 'subject']).size()
per_animal_df['n_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region', 'subject']).size()
per_animal_df['n_FS'] = merged_df[merged_df['type'] == 'FS'].groupby(['full_region', 'subject']).size()
per_animal_df = per_animal_df.reset_index()
per_animal_df['perc_mod_FS'] =  (per_animal_df['modulated_FS'] / per_animal_df['n_FS']) * 100
per_animal_df['perc_mod_RS'] =  (per_animal_df['modulated_RS'] / per_animal_df['n_RS']) * 100
per_animal_df['perc_enh_FS'] =  (per_animal_df['enhanced_FS'] / per_animal_df['n_FS']) * 100
per_animal_df['perc_enh_RS'] =  (per_animal_df['enhanced_RS'] / per_animal_df['n_RS']) * 100
per_animal_df['perc_supp_FS'] =  (per_animal_df['suppressed_FS'] / per_animal_df['n_FS']) * 100
per_animal_df['perc_supp_RS'] =  (per_animal_df['suppressed_RS'] / per_animal_df['n_RS']) * 100
per_animal_df.loc[per_animal_df['n_RS'] < MIN_NEURONS, 'perc_mod_RS'] = np.nan
per_animal_df.loc[per_animal_df['n_FS'] < MIN_NEURONS, 'perc_mod_FS'] = np.nan

# Calculate summary statistics
summary_df = merged_df.groupby(['full_region']).sum()
summary_df['n_neurons'] = merged_df.groupby(['full_region']).size()
summary_df['n_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region']).size()
summary_df['n_FS'] = merged_df[merged_df['type'] == 'FS'].groupby(['full_region']).size()
summary_df['modulation_index_FS'] = merged_df[merged_df['type'] == 'FS'].groupby(['full_region']).mean()['mod_index_late']
summary_df['modulation_index_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_enh_FS'] =  (summary_df['enhanced_FS'] / summary_df['n_FS']) * 100
summary_df['perc_supp_FS'] =  (summary_df['suppressed_FS'] / summary_df['n_FS']) * 100
summary_df['perc_enh_RS'] =  (summary_df['enhanced_RS'] / summary_df['n_RS']) * 100
summary_df['perc_supp_RS'] =  (summary_df['suppressed_RS'] / summary_df['n_RS']) * 100
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['perc_mod_RS'] =  (summary_df['modulated_RS'] / summary_df['n_RS']) * 100
summary_df['perc_mod_FS'] =  (summary_df['modulated_FS'] / summary_df['n_FS']) * 100
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
ordered_regions_FS = summary_df.sort_values('perc_FS_mod', ascending=True).reset_index()

# %% Plot ratio FS/RS modulated neurons

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
sns.barplot(x='100perc', y='full_region', data=summary_df, color=colors['RS'], ax=ax1,
            order=ordered_regions_FS['full_region'], label='RS')
sns.barplot(x='perc_FS_mod', y='full_region', data=summary_df, color=colors['FS'], ax=ax1,
            order=ordered_regions_FS['full_region'], label='FS')
#summary_df[['perc_RS_mod', 'perc_FS_mod']].plot(kind='bar', stacked=True)

ax1.set(ylabel='', xlabel='Fraction of modulated neurons', xticks=[0, 25, 50, 75, 100])
#ax1.plot([0, 0], ax1.get_ylim(), color=colors['grey'], ls='--')
"""
ax1.text(95, -1, 'n =', ha='center', va='center', fontsize=6)
ax1.text(110, -1, 'FS', ha='center', va='center', fontsize=6, fontweight='bold', color=colors['FS'])
ax1.text(125, -1, 'RS', ha='center', va='center', fontsize=6, fontweight='bold', color=colors['RS'])
for i, region_name in enumerate(ordered_regions_FS['full_region']):
    ax1.text(110, i, summary_df.loc[summary_df['full_region'] == region_name, 'n_FS'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
    ax1.text(125, i, summary_df.loc[summary_df['full_region'] == region_name, 'n_RS'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
"""
ax1.text(110, -1, 'Mod. neurons (n)', ha='center', va='center', fontsize=6)
for i, region_name in enumerate(ordered_regions_FS['full_region']):
    ax1.text(110, i, summary_df.loc[summary_df['full_region'] == region_name, 'modulated'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
#ax1.legend(frameon=False, bbox_to_anchor=(0.98, 1))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'ratio_mod_neurons.pdf'))

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi)

sns.barplot(x='perc_mod_RS', y='full_region', data=summary_df, color=colors['RS'], ax=ax1,
            order=ordered_regions_FS['full_region'], label='RS')
ax1.set(ylabel='', xticks=[0, 25, 50, 75], xlabel='% RS neurons')

sns.barplot(x='perc_mod_FS', y='full_region', data=summary_df, color=colors['FS'], ax=ax2,
            order=ordered_regions_FS['full_region'], label='FS')
ax2.set(ylabel='', xticks=[0, 25, 50, 75], yticklabels=[], xlabel='% FS neurons')

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'perc_mod_neurons.pdf'))

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

(
     so.Plot(summary_df, x='perc_mod_FS', y='perc_mod_RS')
     .add(so.Dot(pointsize=2))
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .label(x='Narrow/fast spiking (%)', y='Regular spiking (%)')
     .on(ax1)
     .plot()
)
ax1.set(xticks=[0, 25, 50, 75], yticks=[0, 25, 50, 75])
r, p = pearsonr(summary_df['perc_mod_FS'], summary_df['perc_mod_RS'])

sns.despine(trim=True)
plt.tight_layout()

