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
from scipy.stats import wilcoxon
from os.path import join
from serotonin_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 30

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure3')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'), index_col=0)
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']
neuron_type['neuron_id'] = neuron_type['cluster_id']
merged_df = pd.merge(light_neurons, neuron_type, on=['neuron_id', 'pid', 'eid', 'probe'])
merged_df['merged_region'] = combine_regions(merged_df['region'], abbreviate=True)

# Drop root and void
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['region']) if 'root' in j])
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['region']) if 'void' in j])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Add enhancent and suppressed
merged_df['enhanced_late'] = (merged_df['modulated'] & (merged_df['mod_index_late'] > 0))
merged_df['suppressed_late'] = (merged_df['modulated'] & (merged_df['mod_index_late'] < 0))

# Calculate summary statistics
summary_df = merged_df[merged_df['sert-cre'] == 1].groupby(['type']).sum()
summary_df['n_neurons'] = merged_df[merged_df['sert-cre'] == 1].groupby(['type']).size()
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced_late'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed_late'] / summary_df['n_neurons']) * 100
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = summary_df['perc_supp']

# Summary statistics per mouse
per_mouse_df = merged_df[merged_df['sert-cre'] == 1].groupby(['type', 'subject']).sum()
per_mouse_df['n_neurons'] = merged_df[merged_df['sert-cre'] == 1].groupby(['type', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['modulated'] / per_mouse_df['n_neurons']) * 100
per_mouse_df['perc_enh'] = (per_mouse_df['enhanced_late'] / per_mouse_df['n_neurons']) * 100
per_mouse_df['perc_supp'] = (per_mouse_df['suppressed_late'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df.reset_index()



# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.bar([1.1, 1.9],
        [per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_enh'].mean(),
         per_mouse_df.loc[per_mouse_df['type'] == 'NS', 'perc_enh'].mean()],
        color=[colors['RS'], colors['NS']], width=0.6)
ax1.bar([3.1, 3.9],
        [per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_supp'].mean(),
         per_mouse_df.loc[per_mouse_df['type'] == 'NS', 'perc_supp'].mean()],
        color=[colors['RS'], colors['NS']], width=0.6)
for i, subject in enumerate(np.unique(per_mouse_df['subject'])):
    ax1.plot([1.1, 1.9],
             [per_mouse_df.loc[(per_mouse_df['subject'] == subject) & (per_mouse_df['type'] == 'RS'), 'perc_enh'],
              per_mouse_df.loc[(per_mouse_df['subject'] == subject) & (per_mouse_df['type'] == 'NS'), 'perc_enh']],
             color=colors['grey'])
    ax1.plot([3.1, 3.9],
             [per_mouse_df.loc[(per_mouse_df['subject'] == subject) & (per_mouse_df['type'] == 'RS'), 'perc_supp'],
              per_mouse_df.loc[(per_mouse_df['subject'] == subject) & (per_mouse_df['type'] == 'NS'), 'perc_supp']],
             color=colors['grey'])
_, p_enh = wilcoxon(per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_enh'],
                    per_mouse_df.loc[per_mouse_df['type'] == 'NS', 'perc_enh'])

_, p_supp = wilcoxon(per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_supp'],
                     per_mouse_df.loc[per_mouse_df['type'] == 'NS', 'perc_supp'])
ax1.text(1.5, 38, 'n.s.', ha='center')
ax1.text(3.5, 38, 'n.s.', ha='center')
ax1.set(xticks=[1.5, 3.5], xticklabels=['Enhanced', 'Suppressed'], ylabel='Modulated neurons (%)',
        yticks=[0, 10, 20, 30, 40])
#ax1.legend(frameon=False, bbox_to_anchor=(0.25, 0.8))
sns.despine(trim=False)
plt.tight_layout()
plt.savefig(join(fig_path, 'enh_supp_perc.pdf'))

# %%

per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_enh']
per_mouse_df.loc[per_mouse_df['type'] == 'RS', 'perc_supp']




