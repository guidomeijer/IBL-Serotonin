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
MIN_NEURONS = 20
MIN_PERC = 8

# Paths
fig_path, save_path = paths()
map_path = join(fig_path, 'Ephys', 'BrainMaps')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['full_region'] = combine_regions(all_neurons['region'], split_thalamus=False)

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['expression'] == 1]

# Transform to ms
sert_neurons['latency'] = sert_neurons['latency_peak_hw'] * 1000

# Get percentage modulated per region
reg_neurons = sert_neurons.groupby('full_region').median()['latency'].to_frame()
reg_neurons['n_neurons'] = sert_neurons.groupby(['full_region']).size()
reg_neurons['perc_mod'] = (sert_neurons.groupby(['full_region']).sum()['modulated']
                           / sert_neurons.groupby(['full_region']).size()) * 100
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['full_region'] != 'root']
reg_neurons = reg_neurons[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons[reg_neurons['perc_mod'] >= MIN_PERC]
reg_neurons = reg_neurons.sort_values('latency')

# Apply selection criteria
sert_neurons = sert_neurons[sert_neurons['full_region'].isin(reg_neurons['full_region'])]

# %%

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
sns.pointplot(x='full_region', y='latency', data=sert_neurons, order=reg_neurons['full_region'],
              join=False, ci=68, color=colors['general'], ax=ax1)
ax1.set(ylim=[0, 400], ylabel='5-HT modulation latency (ms)', xlabel='')
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'modulation_latency.png'), dpi=300)
plt.savefig(join(fig_path, 'Ephys', 'modulation_latency.pdf'))

