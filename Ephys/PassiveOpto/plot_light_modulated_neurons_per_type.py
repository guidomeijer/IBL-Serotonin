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
from serotonin_functions import paths, figure_style, load_subjects

# Paths
fig_path, save_path = paths()

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'), index_col=0)
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']
neuron_type['neuron_id'] = neuron_type['cluster_id']
merged_df = pd.merge(light_neurons, neuron_type, on=['neuron_id', 'pid', 'eid', 'probe'])

# Drop ZFM-02180 for now
# light_neurons = light_neurons[light_neurons['subject'] != 'ZFM-02180']

# Drop root and void
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['region']) if 'root' in j])
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.drop(index=[i for i, j in enumerate(merged_df['region']) if 'void' in j])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Add enhancent and suppressed
merged_df['enhanced_late'] = (merged_df['modulated'] & (merged_df['mod_index_late'] > 0))
merged_df['suppressed_late'] = (merged_df['modulated'] & (merged_df['mod_index_late'] < 0))

# Calculate summary statistics
summary_df = merged_df[merged_df['expression'] == 1].groupby(['type']).sum()
summary_df['n_neurons'] = merged_df[merged_df['expression'] == 1].groupby(['type']).size()
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced_late'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed_late'] / summary_df['n_neurons']) * 100
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = summary_df['perc_supp']

summary_no_df = merged_df[merged_df['expression'] == 0].groupby(['type']).sum()
summary_no_df['n_neurons'] = merged_df[merged_df['expression'] == 0].groupby(['type']).size()
summary_no_df = summary_no_df.reset_index()
summary_no_df['perc_enh'] =  (summary_no_df['enhanced_late'] / summary_no_df['n_neurons']) * 100
summary_no_df['perc_supp'] =  (summary_no_df['suppressed_late'] / summary_no_df['n_neurons']) * 100
summary_no_df['ratio'] = summary_no_df['perc_enh'] - summary_no_df['perc_supp']
summary_no_df['perc_supp'] = summary_no_df['perc_supp']


# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=dpi)
"""
ax1.bar(np.arange(2) - 0.15, [summary_df.loc[summary_df['type'] == 'FS', 'perc_enh'].values[0],
                             summary_df.loc[summary_df['type'] == 'RS', 'perc_enh'].values[0]],
        0.3, color=[colors['enhanced'], colors['enhanced']], label='Enhanced')
ax1.bar(np.arange(2) + 0.15, [summary_df.loc[summary_df['type'] == 'FS', 'perc_supp'].values[0],
                             summary_df.loc[summary_df['type'] == 'RS', 'perc_supp'].values[0]],
        0.3, color=[colors['suppressed'], colors['suppressed']], label='Suppressed')
"""
X = [0.5, 1]
ax1.bar(X, [summary_df.loc[summary_df['type'] == 'FS', 'perc_supp'].values[0],
            summary_df.loc[summary_df['type'] == 'RS', 'perc_supp'].values[0]],
        0.3, label='Suppressed', color=colors['suppressed'])

ax1.bar(X, [summary_df.loc[summary_df['type'] == 'FS', 'perc_enh'].values[0],
            summary_df.loc[summary_df['type'] == 'RS', 'perc_enh'].values[0]],
        0.3, label='Enhanced', color=colors['enhanced'],
        bottom=[summary_df.loc[summary_df['type'] == 'FS', 'perc_supp'].values[0],
                summary_df.loc[summary_df['type'] == 'RS', 'perc_supp'].values[0]])


ax1.set(ylabel='5-HT modulated neurons (%)', xticks=X,
        xticklabels=['Fast\nspiking', 'Regular\nspiking'],
        ylim=[0, 50], xlim=[0.2, 1.25])
ax1.legend(frameon=False)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_type.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'light_modulated_neurons_per_type.png'))
