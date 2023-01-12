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
from scipy.stats import pearsonr
from os.path import join
from serotonin_functions import paths, figure_style, load_subjects, combine_regions

# Settings
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure3')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'), index_col=0)
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type[neuron_type['type'] != 'Und.']
neuron_type['neuron_id'] = neuron_type['cluster_id']
merged_df = pd.merge(light_neurons, neuron_type, on=['neuron_id', 'pid', 'eid', 'probe'])
merged_df['full_region'] = combine_regions(merged_df['region'], abbreviate=True)

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_df.loc[merged_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get percentage modulated per region
per_region_df = merged_df.groupby('full_region').mean(numeric_only=True)['mod_index_late'].to_frame()
per_region_df['n_neurons'] = merged_df[merged_df['type'] == 'NS'].groupby(['full_region']).size()
per_region_df['perc_mod_NS'] = (merged_df[merged_df['type'] == 'NS'].groupby(['full_region']).sum()['modulated']
                                / merged_df[merged_df['type'] == 'NS'].groupby(['full_region']).size()) * 100
per_region_df = per_region_df.reset_index()
per_region_df = per_region_df[per_region_df['full_region'] != 'root']
per_region_df = per_region_df[per_region_df['full_region'] != 'ZI']
per_region_df = per_region_df[per_region_df['n_neurons'] >= MIN_NEURONS]


# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
per_region_df['color'] = [colors[i] for i in per_region_df['full_region']]

(
     so.Plot(per_region_df, x='mod_index_late', y='perc_mod_NS')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)
for i in per_region_df.index:
    ax1.text(per_region_df.loc[i, 'mod_index_late'] ,
             per_region_df.loc[i, 'perc_mod_NS'],
             per_region_df.loc[i, 'full_region'],
             ha='center', va='center',
             color=per_region_df.loc[i, 'color'], fontsize=4.5, fontweight='bold')
ax1.set(yticks=[0, 25, 50], xticks=[-0.06, -0.03, 0, 0.03],
        ylabel='Modulated inhibitory neurons (%)', xlabel='Modulation index')
r, p = pearsonr(per_region_df['mod_index_late'], per_region_df['perc_mod_NS'])
#ax1.text(0.1, 100, f'r = {r:.2f}', fontsize=6)
#ax1.text(-0.1, 520, '***', fontsize=10, ha='center')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_vs_perc_mod_NS.pdf'))

