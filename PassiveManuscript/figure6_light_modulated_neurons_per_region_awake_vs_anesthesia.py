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
from scipy.stats import wilcoxon
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS_POOLED = 5
MIN_NEURONS_PER_MOUSE = 1
MIN_MOD_NEURONS = 1
MIN_REC = 1

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure6')

# Load in results
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))
light_neurons = pd.merge(awake_neurons, anes_neurons, on=['pid', 'eid', 'subject', 'neuron_id',
                                                        'date', 'probe', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'])
light_neurons['abr_region'] = combine_regions(light_neurons['region'], abbreviate=True)

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Get modulated neurons
mod_neurons = light_neurons[(light_neurons['sert-cre'] == 1)
                            & ((light_neurons['modulated_x'] == 1) | (light_neurons['modulated_y'] == 1))]
mod_neurons = mod_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)

# Get region statistics
per_region = mod_neurons.groupby(['full_region', 'abr_region']).median(numeric_only=True).reset_index()

# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], color='k', ls='--', zorder=0)
#ax1.plot([-0.3, 0.3], [0, 0], color='k', ls='--', zorder=0)
(
 so.Plot(per_region, x='mod_index_late_x', y='mod_index_late_y')
 .add(so.Dot(pointsize=3))
 .label(x='Modulation index awake', y='Modulation index anesthesia')
 .on(ax1)
 .plot()
 )
_, p = wilcoxon(per_region['mod_index_late_x'], per_region['mod_index_late_y'])
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
per_region['color'] = [colors[i] for i in per_region['full_region']]

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], color='k', ls='--', zorder=0)
(
 so.Plot(per_region, x='mod_index_late_x', y='mod_index_late_y')
 .add(so.Dot(pointsize=0))
 .on(ax1)
 .plot()
 )
for i in per_region.index:
    ax1.text(per_region.loc[i, 'mod_index_late_x'],
             per_region.loc[i, 'mod_index_late_y'],
             per_region.loc[i, 'abr_region'],
             ha='center', va='center',
             color=per_region.loc[i, 'color'], fontsize=4.5, fontweight='bold')
_, p = wilcoxon(per_region['mod_index_late_x'], per_region['mod_index_late_y'])
if p < 0.01:
    ax1.text(0, 0.25, '**', fontsize=12, ha='center', va='center')
elif p < 0.05:
    ax1.text(0, 0.25, '*', fontsize=12, ha='center', va='center')
ax1.set(xticks=np.arange(-0.3, 0.4, 0.3), yticks=np.arange(-0.3, 0.4, 0.3),
        xlabel='Modulation index awake', ylabel='Modulation index anesthesia')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'per_region_mod_index_awake_vs_anesthesia.pdf'))



