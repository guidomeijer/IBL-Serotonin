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
from scipy.stats import ttest_ind
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS = 3
MIN_REC = 3

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

# Drop neurons that could not be defined as RS or NS
merged_df = merged_df[(merged_df['type'] == 'RS') | (merged_df['type'] == 'NS')]

# Drop root and void
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['full_region']) if 'root' in j])
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['full_region']) if 'void' in j])

# Add enhanced and suppressed
merged_df['modulated_NS'] = merged_df['modulated'] & (merged_df['type'] == 'NS')
merged_df['modulated_RS'] = merged_df['modulated'] & (merged_df['type'] == 'RS')
merged_df['enhanced_NS'] = merged_df['modulated'] & (merged_df['mod_index_late'] > 0) & (merged_df['type'] == 'NS')
merged_df['suppressed_NS'] = merged_df['modulated'] & (merged_df['mod_index_late'] < 0) & (merged_df['type'] == 'NS')
merged_df['enhanced_RS'] = merged_df['modulated'] & (merged_df['mod_index_late'] > 0) & (merged_df['type'] == 'RS')
merged_df['suppressed_RS'] = merged_df['modulated'] & (merged_df['mod_index_late'] < 0) & (merged_df['type'] == 'RS')

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
merged_df = merged_df[merged_df['sert-cre'] == 1]

# Calculate stats per animal
per_animal_df = merged_df.groupby(['full_region', 'subject']).sum(numeric_only=True)
per_animal_df['n_neurons'] = merged_df.groupby(['full_region', 'subject']).size()
per_animal_df['n_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region', 'subject']).size()
per_animal_df['n_NS'] = merged_df[merged_df['type'] == 'NS'].groupby(['full_region', 'subject']).size()
per_animal_df.loc[np.isnan(per_animal_df['n_NS']), 'n_NS'] = 0
per_animal_df.loc[np.isnan(per_animal_df['n_RS']), 'n_RS'] = 0
per_animal_df = per_animal_df.reset_index()
per_animal_df['perc_mod_NS'] =  (per_animal_df['modulated_NS'] / per_animal_df['n_NS']) * 100
per_animal_df['perc_mod_RS'] =  (per_animal_df['modulated_RS'] / per_animal_df['n_RS']) * 100
per_animal_df['perc_enh_NS'] =  (per_animal_df['enhanced_NS'] / per_animal_df['n_NS']) * 100
per_animal_df['perc_enh_RS'] =  (per_animal_df['enhanced_RS'] / per_animal_df['n_RS']) * 100
per_animal_df['perc_supp_NS'] =  (per_animal_df['suppressed_NS'] / per_animal_df['n_NS']) * 100
per_animal_df['perc_supp_RS'] =  (per_animal_df['suppressed_RS'] / per_animal_df['n_RS']) * 100
per_animal_df = per_animal_df[per_animal_df['n_RS'] >= MIN_NEURONS]
per_animal_df = per_animal_df[per_animal_df['n_NS'] >= MIN_NEURONS]
per_animal_df = per_animal_df.groupby('full_region').filter(lambda x: len(x) >= MIN_REC)

# Calculate summary statistics
summary_df = merged_df.groupby(['full_region']).sum()
summary_df['n_neurons'] = merged_df.groupby(['full_region']).size()
summary_df['n_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region']).size()
summary_df['n_NS'] = merged_df[merged_df['type'] == 'NS'].groupby(['full_region']).size()
summary_df['modulation_index_NS'] = merged_df[merged_df['type'] == 'NS'].groupby(['full_region']).mean()['mod_index_late']
summary_df['modulation_index_RS'] = merged_df[merged_df['type'] == 'RS'].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_enh_NS'] =  (summary_df['enhanced_NS'] / summary_df['n_NS']) * 100
summary_df['perc_supp_NS'] =  (summary_df['suppressed_NS'] / summary_df['n_NS']) * 100
summary_df['perc_enh_RS'] =  (summary_df['enhanced_RS'] / summary_df['n_RS']) * 100
summary_df['perc_supp_RS'] =  (summary_df['suppressed_RS'] / summary_df['n_RS']) * 100
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['perc_mod_RS'] =  (summary_df['modulated_RS'] / summary_df['n_RS']) * 100
summary_df['perc_mod_NS'] =  (summary_df['modulated_NS'] / summary_df['n_NS']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['perc_supp_NS'] = -summary_df['perc_supp_NS']
summary_df['perc_supp_RS'] = -summary_df['perc_supp_RS']

# Get ordered regions by percentage modulated
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# Calculate percentage of NS and RS modulated neurons per region
summary_df['perc_RS_mod'] = (summary_df['perc_mod_RS'] /
                             (summary_df['perc_mod_RS'] + summary_df['perc_mod_NS'])) * 100
summary_df['perc_NS_mod'] = (summary_df['perc_mod_NS'] /
                             (summary_df['perc_mod_RS'] + summary_df['perc_mod_NS'])) * 100
summary_df['100perc'] = 100

# Get ordered regions by ratio NS/NS
ordered_regions_NS = summary_df.sort_values('perc_NS_mod', ascending=True).reset_index()

# %% Plot ratio NS/RS modulated neurons

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
sns.barplot(x='100perc', y='full_region', data=summary_df, color=colors['RS'], ax=ax1,
            order=ordered_regions_NS['full_region'], label='RS')
sns.barplot(x='perc_NS_mod', y='full_region', data=summary_df, color=colors['NS'], ax=ax1,
            order=ordered_regions_NS['full_region'], label='NS')
#summary_df[['perc_RS_mod', 'perc_NS_mod']].plot(kind='bar', stacked=True)

ax1.set(ylabel='', xlabel='Fraction of modulated neurons', xticks=[0, 25, 50, 75, 100])
#ax1.plot([0, 0], ax1.get_ylim(), color=colors['grey'], ls='--')
"""
ax1.text(95, -1, 'n =', ha='center', va='center', fontsize=6)
ax1.text(110, -1, 'NS', ha='center', va='center', fontsize=6, fontweight='bold', color=colors['NS'])
ax1.text(125, -1, 'RS', ha='center', va='center', fontsize=6, fontweight='bold', color=colors['RS'])
for i, region_name in enumerate(ordered_regions_NS['full_region']):
    ax1.text(110, i, summary_df.loc[summary_df['full_region'] == region_name, 'n_NS'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
    ax1.text(125, i, summary_df.loc[summary_df['full_region'] == region_name, 'n_RS'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
"""
ax1.text(110, -1, 'Mod. neurons (n)', ha='center', va='center', fontsize=6)
for i, region_name in enumerate(ordered_regions_NS['full_region']):
    ax1.text(110, i, summary_df.loc[summary_df['full_region'] == region_name, 'modulated'].values[0].astype(int),
             va='center', ha='center', fontsize=6)
#ax1.legend(frameon=False, bbox_to_anchor=(0.98, 1))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'ratio_mod_neurons.pdf'))

# %%

reg_order = ['Medial prefrontal cortex', 'Orbitofrontal cortex',
             'Secondary motor cortex', 'Retrosplenial cortex', 'Visual cortex',
             'Piriform', 'Tail of the striatum', 'Hippocampus']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi)

sns.barplot(x='perc_mod_RS', y='full_region', data=summary_df, color=colors['RS'], ax=ax1,
            label='RS')
ax1.set(ylabel='', xticks=[0, 25, 50, 75], xlabel='% RS neurons')

sns.barplot(x='perc_mod_NS', y='full_region', data=summary_df, color=colors['NS'], ax=ax2,
            label='NS')
ax2.set(ylabel='', xticks=[0, 25, 50, 75], yticklabels=[], xlabel='% NS neurons')

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'perc_mod_neurons.pdf'))

# %%

per_animal_long_df = pd.melt(per_animal_df, id_vars=['subject', 'full_region'],
                             value_vars=['perc_mod_RS', 'perc_mod_NS'])

# Do statistics
p_vals = dict()
for i, region in enumerate(np.unique(per_animal_df['full_region'])):
    p_vals[region] = ttest_ind(per_animal_df.loc[per_animal_df['full_region'] == region, 'perc_mod_NS'],
                               per_animal_df.loc[per_animal_df['full_region'] == region, 'perc_mod_RS'],
                               nan_policy='omit')[1]

f, ax1 = plt.subplots(1, 1, figsize=(3, 2.25), dpi=dpi)

bplt = sns.barplot(x='value', y='full_region', data=per_animal_long_df, color=colors['RS'], ax=ax1,
                   errorbar='se', hue='variable', hue_order=['perc_mod_NS', 'perc_mod_RS'],
                   order=reg_order, palette=[colors['NS'], colors['RS']], errwidth=1)
ax1.set(ylabel='', xticks=[0, 25, 50, 75], xlabel='Modulated neurons (%)')
ax1.legend(frameon=False, bbox_to_anchor=(0.6, 0.5))
new_labels = ['NS', 'RS']
for t, l in zip(bplt.legend_.texts, new_labels):
    t.set_text(l)

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'perc_mod_neurons_NS_RS.pdf'))

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2), dpi=dpi)

sns.barplot(x='perc_supp_NS', y='full_region', data=summary_df, color=colors['suppressed'], ax=ax1,
            order=reg_order)
sns.barplot(x='perc_enh_NS', y='full_region', data=summary_df, color=colors['enhanced'], ax=ax1,
            order=reg_order)
ax1.set(ylabel='', xticks=[-75, -50, -25, 0, 25, 50], xticklabels=[75, 50, 25, 0, 25, 50],
        xlabel='')
ax1.set_title('NS', fontweight='bold')

sns.barplot(x='perc_supp_RS', y='full_region', data=summary_df, color=colors['suppressed'], ax=ax2,
            order=reg_order)
sns.barplot(x='perc_enh_RS', y='full_region', data=summary_df, color=colors['enhanced'], ax=ax2,
            order=reg_order)
ax2.set(ylabel='', xticks=[-75, -50, -25, 0, 25, 50], xticklabels=[75, 50, 25, 0, 25, 50],
        title='RS', yticklabels=[], xlabel='')
ax2.set_title('RS', fontweight='bold')

f.text(0.7, 0.03, '% modulated neurons', ha='center', va='center')

sns.despine(trim=True)
plt.tight_layout(pad=1.5)

plt.savefig(join(fig_path, 'perc_mod_neurons_signed.pdf'))



