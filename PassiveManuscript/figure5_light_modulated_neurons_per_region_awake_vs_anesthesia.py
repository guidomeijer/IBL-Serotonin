#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import wilcoxon, ttest_rel
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_MOD_NEURONS = 3

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in results
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    awake_neurons.loc[awake_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    anes_neurons.loc[anes_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get summary per region
awake_neurons['full_region'] = combine_regions(awake_neurons['region'])
awake_neurons['abr_region'] = combine_regions(awake_neurons['region'], abbreviate=True)
awake_neurons = awake_neurons[(awake_neurons['sert-cre'] == 1) & (awake_neurons['modulated'] == 1)]
awake_neurons = awake_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)
awake_regions = awake_neurons.groupby(['full_region', 'abr_region']).mean(numeric_only=True).reset_index()

anes_neurons['full_region'] = combine_regions(anes_neurons['region'])
anes_neurons['abr_region'] = combine_regions(anes_neurons['region'], abbreviate=True)
anes_neurons = anes_neurons[(anes_neurons['sert-cre'] == 1) & (anes_neurons['modulated'] == 1)]
anes_neurons = anes_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)
anes_regions = anes_neurons.groupby(['full_region', 'abr_region']).mean(numeric_only=True).reset_index()

# Merge two dataframes
merged_regions = pd.merge(awake_regions, anes_regions, on=['full_region', 'abr_region'])

# Drop root
merged_regions = merged_regions[merged_regions['full_region'] != 'root']

# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], color='lightgrey', ls='--', zorder=0)
#ax1.plot([-0.3, 0.3], [0, 0], color='k', ls='--', zorder=0)
(
 so.Plot(merged_regions, x='mod_index_late_x', y='mod_index_late_y')
 .add(so.Dot(pointsize=3))
 .label(x='Modulation index awake', y='Modulation index anesthesia')
 .on(ax1)
 .plot()
 )
_, p = ttest_rel(merged_regions['mod_index_late_x'], merged_regions['mod_index_late_y'])
if p < 0.01:
    ax1.text(0, 0.25, '**', fontsize=12, ha='center', va='center')
elif p < 0.05:
    ax1.text(0, 0.25, '*', fontsize=12, ha='center', va='center')
ax1.set(xticks=np.arange(-0.3, 0.4, 0.3), yticks=np.arange(-0.3, 0.4, 0.3))
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'per_region_mod_index_awake_vs_anesthesia_dots.pdf'))

# %% Plot percentage modulated neurons per region

# Add colormap
merged_regions['color'] = [colors[i] for i in merged_regions['full_region']]

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], color='lightgrey', ls='--', zorder=0)
(
 so.Plot(merged_regions, x='mod_index_late_x', y='mod_index_late_y')
 .add(so.Dot(pointsize=0))
 .on(ax1)
 .plot()
 )
for i in merged_regions.index:
    ax1.text(merged_regions.loc[i, 'mod_index_late_x'],
             merged_regions.loc[i, 'mod_index_late_y'],
             merged_regions.loc[i, 'abr_region'],
             ha='center', va='center',
             color=merged_regions.loc[i, 'color'], fontsize=4.5, fontweight='bold')
_, p = ttest_rel(merged_regions['mod_index_late_x'], merged_regions['mod_index_late_y'])
if p < 0.01:
    ax1.text(0, 0.25, '**', fontsize=12, ha='center', va='center')
elif p < 0.05:
    ax1.text(0, 0.25, '*', fontsize=12, ha='center', va='center')
ax1.set(xticks=np.arange(-0.3, 0.4, 0.3), yticks=np.arange(-0.3, 0.4, 0.3),
        xlabel='Modulation index awake', ylabel='Modulation index anesthesia')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'per_region_mod_index_awake_vs_anesthesia.pdf'))



