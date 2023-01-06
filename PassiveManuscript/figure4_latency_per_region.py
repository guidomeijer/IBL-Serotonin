#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from os.path import join
from matplotlib.colors import ListedColormap
from serotonin_functions import (paths, figure_style, load_subjects, plot_scalar_on_slice,
                                 combine_regions)

# Settings
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['full_region'] = combine_regions(all_neurons['region'])
all_neurons['abr_region'] = combine_regions(all_neurons['region'], abbreviate=True)

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only modulated neurons in sert-cre mice
sert_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 1)]
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

# Get absolute
sert_neurons['mod_index_abs'] = sert_neurons['mod_index_late'].abs()

# Group by region
grouped_df = sert_neurons.groupby(['abr_region', 'full_region']).median(numeric_only=True).reset_index().reset_index()

# Drop root
grouped_df = grouped_df[grouped_df['abr_region'] != 'root']

# Convert to ms
grouped_df['latency'] = grouped_df['latency'] * 1000
sert_neurons['latency'] = sert_neurons['latency'] * 1000


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
ax1.set(xlabel='Modulation onset latency (ms)', ylabel='', xticks=[0, 500, 1000], xlim=[-150, 1150])
#plt.xticks(rotation=90)
for i, region in enumerate(ordered_regions['full_region']):
    this_lat = ordered_regions.loc[ordered_regions['full_region'] == region, 'latency'].values[0] * 1000
    ax1.text(1200, i+0.25, f'{this_lat:.0f} ms', fontsize=5)
plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'modulation_latency_per_region.pdf'))

# %%

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]
newcmp = ListedColormap(grouped_df['color'])

# Add text alignment
grouped_df['ha'] = 'right'
grouped_df.loc[grouped_df['abr_region'] == 'Amyg', 'ha'] = 'left'
grouped_df.loc[grouped_df['abr_region'] == 'mPFC', 'ha'] = 'left'
grouped_df.loc[grouped_df['abr_region'] == 'RSP', 'ha'] = 'left'
grouped_df.loc[grouped_df['abr_region'] == 'Str', 'ha'] = 'left'
grouped_df.loc[grouped_df['abr_region'] == 'MRN', 'ha'] = 'left'
grouped_df.loc[grouped_df['abr_region'] == 'PPC', 'ha'] = 'center'
grouped_df.loc[grouped_df['abr_region'] == 'SC', 'ha'] = 'center'
grouped_df['va'] = 'bottom'
grouped_df.loc[grouped_df['abr_region'] == 'M2', 'va'] = 'top'
grouped_df.loc[grouped_df['abr_region'] == 'RSP', 'va'] = 'top'
grouped_df.loc[grouped_df['abr_region'] == 'Amyg', 'va'] = 'top'
grouped_df.loc[grouped_df['abr_region'] == 'Str', 'va'] = 'center'
grouped_df['x_offset'] = 0
grouped_df.loc[grouped_df['abr_region'] == 'Amyg', 'x_offset'] = 0.01
grouped_df.loc[grouped_df['abr_region'] == 'Str', 'x_offset'] = 0.01
grouped_df.loc[grouped_df['abr_region'] == 'mPFC', 'x_offset'] = 0.01
grouped_df.loc[grouped_df['abr_region'] == 'MRN', 'x_offset'] = 0.01
grouped_df.loc[grouped_df['abr_region'] == 'RSP', 'x_offset'] = 0.01
grouped_df.loc[grouped_df['abr_region'] == 'PPC', 'x_offset'] = 0.015
grouped_df.loc[grouped_df['abr_region'] == 'PAG', 'x_offset'] = -0.01
grouped_df.loc[grouped_df['abr_region'] == 'Thal', 'x_offset'] = -0.01
grouped_df.loc[grouped_df['abr_region'] == 'Pir', 'x_offset'] = -0.01
grouped_df.loc[grouped_df['abr_region'] == 'Hipp', 'x_offset'] = -0.01
grouped_df.loc[grouped_df['abr_region'] == 'M2', 'x_offset'] = -0.01
grouped_df['y_offset'] = 0
grouped_df.loc[grouped_df['abr_region'] == 'PPC', 'y_offset'] = 5
grouped_df.loc[grouped_df['abr_region'] == 'SC', 'y_offset'] = 5
grouped_df.loc[grouped_df['abr_region'] == 'RSP', 'y_offset'] = -5


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
(
     so.Plot(grouped_df, x='mod_index_late', y='latency', color='index')
     .add(so.Dot(pointsize=2, edgecolor='w', edgewidth=0.5))
     .add(so.Line(color='k', linewidth=1, linestyle='--'), so.PolyFit(order=1))
     .scale(color=newcmp)
     .label(x='Modulation index', y='Modulation latency (ms)')
     .on(ax1)
     .plot()
)
for i in grouped_df.index:
    ax1.text(grouped_df.loc[i, 'mod_index_late'] + grouped_df.loc[i, 'x_offset'],
             grouped_df.loc[i, 'latency'] + grouped_df.loc[i, 'y_offset'],
             grouped_df.loc[i, 'abr_region'],
             ha=grouped_df.loc[i, 'ha'], va=grouped_df.loc[i, 'va'],
             color=grouped_df.loc[i, 'color'], fontsize=4.5, fontweight='bold')
ax1.set(yticks=[0, 200, 400, 600], xticks=[-0.4, -0.2, 0, 0.2])
r, p = pearsonr(grouped_df['mod_index_late'], grouped_df['latency'])
ax1.text(-0.35, 520, f'r = {r:.2f}', fontsize=6)

(
     so.Plot(grouped_df, x='mod_index_abs', y='latency')
     .add(so.Dot(pointsize=2.5))
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .label(x='Absolute modulation index', y='Modulation latency (ms)')
     .on(ax2)
     .plot()
)
ax2.set(yticks=[0, 200, 400, 600], xticks=[0, 0.25, 0.5])

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_latency_vs_index_points.pdf'))


# %%

# Add colormap
grouped_df['color'] = [colors[i] for i in grouped_df['full_region']]

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
     so.Plot(grouped_df, x='mod_index_late', y='latency')
     .add(so.Dot(pointsize=0))
     .add(so.Line(color='grey', linewidth=1), so.PolyFit(order=1))
     .on(ax1)
     .plot()
)
for i in grouped_df.index:
    ax1.text(grouped_df.loc[i, 'mod_index_late'] ,
             grouped_df.loc[i, 'latency'],
             grouped_df.loc[i, 'abr_region'],
             ha='center', va='center',
             color=grouped_df.loc[i, 'color'], fontsize=4.5, fontweight='bold')
ax1.set(yticks=[0, 200, 400, 600], xticks=[-0.4, -0.2, 0, 0.2],
        ylabel='Modulation latency (ms)', xlabel='Modulation index')
r, p = pearsonr(grouped_df['mod_index_late'], grouped_df['latency'])
#ax1.text(0.1, 100, f'r = {r:.2f}', fontsize=6)
ax1.text(-0.1, 520, '***', fontsize=10, ha='center')

sns.despine(offset=2, trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_latency_vs_index.pdf'))



