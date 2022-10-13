#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import (paths, figure_style, load_subjects, plot_scalar_on_slice,
                                 combine_regions)

# Settings
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['full_region'] = combine_regions(all_neurons['region'], split_thalamus=False)

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]

# Transform to ms
sert_neurons['latency'] = sert_neurons['latency_peak_onset']

# Get percentage modulated per region
reg_neurons = sert_neurons.groupby('full_region').median()['latency'].to_frame()
reg_neurons['n_neurons'] = sert_neurons.groupby(['full_region']).size()
reg_neurons['perc_mod'] = (sert_neurons.groupby(['full_region']).sum()['modulated']
                           / sert_neurons.groupby(['full_region']).size()) * 100
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['full_region'] != 'root']
reg_neurons = reg_neurons[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.sort_values('latency')

# Apply selection criteria
sert_neurons = sert_neurons[sert_neurons['full_region'].isin(reg_neurons['full_region'])]
sert_neurons.loc[sert_neurons['latency'] == 0, 'latency'] = np.nan

# Order regions
ordered_regions = sert_neurons.groupby('full_region').median().sort_values('latency', ascending=True).reset_index()

# Convert to log scale
sert_neurons['log_latency'] = np.log10(sert_neurons['latency'])

# %%

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
#sns.pointplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#              join=False, ci=68, color=colors['general'], ax=ax1)
#sns.boxplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#            color=colors['general'], fliersize=0, linewidth=0.75, ax=ax1)
sns.violinplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
               color=colors['grey'], linewidth=0, ax=ax1)
sns.stripplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
               color='k', size=1, ax=ax1)
ax1.set(xlabel='Modulation onset latency (s)', ylabel='', xticks=[0, 0.5, 1], xlim=[-0.15, 1.15])
#plt.xticks(rotation=90)
for i, region in enumerate(ordered_regions['full_region']):
    this_lat = ordered_regions.loc[ordered_regions['full_region'] == region, 'latency'].values[0] * 1000
    ax1.text(1.2, i+0.25, f'{this_lat:.0f} ms', fontsize=5)
plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'modulation_latency_per_region.pdf'))

