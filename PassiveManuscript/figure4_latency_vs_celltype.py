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
import seaborn.objects as so
from scipy.stats import mannwhitneyu
from serotonin_functions import (paths, figure_style, load_subjects, plot_scalar_on_slice,
                                 combine_regions, high_level_regions)

# Settings
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))
neuron_type = neuron_type.rename(columns={'cluster_id': 'neuron_id'})
all_neurons = pd.merge(light_neurons, neuron_type, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])
all_neurons['full_region'] = combine_regions(all_neurons['region'], split_thalamus=False)
all_neurons['high_region'] = high_level_regions(all_neurons['region'])

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]

# Transform to ms
sert_neurons['latency'] = sert_neurons['latency_peak_onset'] * 1000

# Do some filtering
sert_neurons = sert_neurons[sert_neurons['type'] != 'Und.']
sert_neurons = sert_neurons[~np.isnan(sert_neurons['latency'])]

# %%

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)

sns.boxplot(x='type', y='latency', data=sert_neurons[sert_neurons['type'] != 'Und.'], ax=ax1,
            palette=[colors['RS'], colors['NS']], width=0.7)
_, p = mannwhitneyu(sert_neurons.loc[sert_neurons['type'] == 'RS', 'latency'],
                    sert_neurons.loc[sert_neurons['type'] == 'NS', 'latency'])
ax1.set(xlabel='', ylabel='Modulation latency (ms)', xticklabels=['RS', 'NS'],
        yticks=[0, 250, 500, 750, 1000])
ax1.text(0.5, 1100, 'n.s.', ha='center', va='center')

plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'modulation_vs_celltype.pdf'))

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)

sns.boxplot(x='type', y='latency', data=sert_neurons[(sert_neurons['type'] != 'Und.')
                                                     & (sert_neurons['full_region'] == 'Visual cortex')], ax=ax1,
            palette=[colors['RS'], colors['NS']], width=0.7)
_, p = mannwhitneyu(sert_neurons.loc[sert_neurons['type'] == 'RS', 'latency'],
                    sert_neurons.loc[sert_neurons['type'] == 'NS', 'latency'])
ax1.set(xlabel='', ylabel='Modulation latency (ms)', xticklabels=['RS', 'NS'],
        yticks=[0, 250, 500, 750, 1000])
ax1.text(0.5, 1100, 'n.s.', ha='center', va='center')

plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'modulation_vs_celltype_visual.pdf'))





