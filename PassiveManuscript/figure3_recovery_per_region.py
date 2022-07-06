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
from matplotlib.patches import Rectangle
from serotonin_functions import (paths, figure_style, load_subjects, plot_scalar_on_slice,
                                 combine_regions)

# Settings
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure3')

# Load in results
all_neurons = pd.read_pickle(join(save_path, 'recovery_tau.pickle'))
all_neurons['full_region'] = combine_regions(all_neurons['acronym'], split_thalamus=False)

# Exclude outliers
all_neurons = all_neurons[all_neurons['tau'] < 5]

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Select regions with enough neurons
sert_neurons = sert_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_NEURONS)

# Order regions
ordered_regions = sert_neurons.groupby('full_region').median().sort_values('tau', ascending=True).reset_index()


# %%

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2.2), dpi=dpi)
#sns.pointplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#              join=False, ci=68, color=colors['general'], ax=ax1)
#sns.boxplot(x='latency', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
#            color=colors['general'], fliersize=0, linewidth=0.75, ax=ax1)
sns.violinplot(x='tau', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
               color=colors['grey'], linewidth=0, ax=ax1)
sns.stripplot(x='tau', y='full_region', data=sert_neurons, order=ordered_regions['full_region'],
               color='k', size=1, ax=ax1)
ax1.set(xlabel='Modulation recovery time (tau [s])', ylabel='', xlim=[0, 5])
#plt.xticks(rotation=90)
for i, region in enumerate(ordered_regions['full_region']):
    this_lat = ordered_regions.loc[ordered_regions['full_region'] == region, 'tau'].values[0]
    ax1.text(5, i+0.25, f'{this_lat:.1f} s', fontsize=5)
plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'recovery_time_per_region.pdf'))

# %%
long_df = pd.DataFrame()
for i in sert_neurons.index:
    long_df = pd.concat((long_df, pd.DataFrame(data={
        'peth_ratio': sert_neurons.loc[i, 'peth_ratio'],
        'time': sert_neurons.loc[i, 'time']})), ignore_index=True)

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.3, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='peth_ratio', data=long_df, ax=ax1, ci=68, color='k')
ax1.set(xlabel='Time (s)', ylabel='Absolute ratio FR change', xticks=[-1, 0, 1, 2, 3, 4])
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'abs_ratio_fr_change_all_neurons.pdf'))




