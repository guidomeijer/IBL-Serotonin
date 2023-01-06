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
from scipy.stats import pearsonr
import seaborn.objects as so
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
sert_neurons['latency'] = sert_neurons['latency'] * 1000

# Get absolute
sert_neurons['mod_index_abs'] = sert_neurons['mod_index_late'].abs()

# %%

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

(
     so.Plot(sert_neurons, x='mod_index_late', y='latency')
     .add(so.Dot(pointsize=2))
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .limit(x=[-1, 1], y=[-15, 1000])
     .label(x='Modulation index', y='Modulation latency (ms)')
     .on(ax1)
     .plot()
)

(
     so.Plot(sert_neurons, x='mod_index_abs', y='latency')
     .add(so.Dot(pointsize=2))
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .limit(x=[-0.1, 1], y=[-15, 1000])
     .label(x='Absolute modulation index', y='Modulation latency (ms)')
     .on(ax2)
     .plot()
)
ax1.set(title='All neurons')

plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'modulation_vs_latency.pdf'))


# %% Medial prefrontal cortex

df_slice = sert_neurons[(sert_neurons['full_region'] == 'Medial prefrontal cortex')
                        & (sert_neurons['type'] != 'Und.')]
df_slice = df_slice[~np.isnan(df_slice['latency'])]
r, p = pearsonr(df_slice['mod_index_late'], df_slice['latency'])
print(f'r = {r:.2f}\np = {p}')
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
     so.Plot(df_slice, x='mod_index_late', y='latency')
     .add(so.Dot(pointsize=2), color='type')
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .scale(color=[colors['RS'], colors['FS']])
     .limit(x=[-1, 1], y=[-15, 1000])
     .label(x='Modulation index', y='Modulation latency (ms)')
     .on(ax1)
     .plot()
)
ax1.text(0, 1000, '***', fontsize=10, ha='center')
legend = f.legends.pop(0)
ax1.legend(legend.legendHandles, ['RS', 'NS'], frameon=False,
           prop={'size': 6}, handletextpad=0, bbox_to_anchor=[0.3, 1])

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'modulation_vs_latency_frontal.pdf'))

# %% Visual cortex

df_slice = sert_neurons[(sert_neurons['full_region'] == 'Visual cortex')
                        & (sert_neurons['type'] != 'Und.')]
df_slice = df_slice[~np.isnan(df_slice['latency'])]
r, p = pearsonr(df_slice['mod_index_late'], df_slice['latency'])
print(f'r = {r:.2f}\np = {p}')
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
     so.Plot(df_slice, x='mod_index_late', y='latency')
     .add(so.Dot(pointsize=2), color='type')
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .scale(color=[colors['RS'], colors['FS']])
     .limit(x=[-0.6, 0.6], y=[-15, 1000])
     .label(x='Modulation index', y='Modulation latency (ms)')
     .on(ax1)
     .plot()
)
ax1.text(0, 1000, '***', fontsize=10, ha='center')
legend = f.legends.pop(0)
ax1.legend(legend.legendHandles, ['RS', 'NS'], frameon=False,
           prop={'size': 6}, handletextpad=0, bbox_to_anchor=[0.3, 1])

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'modulation_vs_latency_visual.pdf'))






