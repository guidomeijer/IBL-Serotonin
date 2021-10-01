#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:19:11 2021
By: Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from serotonin_functions import paths, get_full_region_name, figure_style

# Settings
_, fig_path, save_path = paths()
PRE_TIME = 0
POST_TIME = 0.3
MIN_IMPROVEMENT = 5

# Load in results
decoding_result = pd.read_csv(join(save_path, 'lda_decoding_probe_trials.csv'))

# Calculate delta
decoding_result['delta_block_on'] = (decoding_result['acc_light_on_block_on'] - decoding_result['acc_light_off_block_on']) * 100
decoding_result['delta_block_off'] = (decoding_result['acc_light_on_block_off'] - decoding_result['acc_light_off_block_off']) * 100

# Get full region names
decoding_result['full_region'] = get_full_region_name(decoding_result['region'])

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'mean_on'] = decoding_result.loc[decoding_result['region'] == region, 'delta_block_on'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'mean_off'] = decoding_result.loc[decoding_result['region'] == region, 'delta_block_off'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(decoding_result['region'] == region)


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

decoding_plot = decoding_result[(decoding_result['sert-cre'] == 1) & (decoding_result['mean_on'].abs() >= MIN_IMPROVEMENT)]
sort_regions = decoding_plot.groupby('full_region').max().sort_values('mean_on', ascending=False).reset_index()['full_region']
sns.barplot(x='mean_on', y='full_region', data=decoding_plot, order=sort_regions,
            ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2", ax=ax1)
sns.swarmplot(x='delta_block_on', y='full_region', data=decoding_plot, order=sort_regions, size=3.5, ax=ax1)

ax1.set(xlabel='Stimulation induced decoding improvement of prior (% correct)', ylabel='', xlim=[-50, 50])

decoding_plot = decoding_result[(decoding_result['sert-cre'] == 1) & (decoding_result['mean_off'].abs() >= MIN_IMPROVEMENT)]
sort_regions = decoding_plot.groupby('full_region').max().sort_values('mean_off', ascending=False).reset_index()['full_region']
sns.barplot(x='mean_off', y='full_region', data=decoding_plot, order=sort_regions,
            ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2", ax=ax2)
sns.swarmplot(x='delta_block_off', y='full_region', data=decoding_plot, order=sort_regions, size=3.5, ax=ax2)

ax2.set(xlabel='Stimulation induced decoding improvement of prior (% correct)', ylabel='', xlim=[-50, 50])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'LDA', f'lda_opto_improvement_probe_trials.png'))

f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
ax1.plot([-100, 100], [0, 0], ls='--', color='gray')
ax1.plot([0, 0], [-100, 100], ls='--', color='gray')
sns.scatterplot(x='mean_on', y='mean_off', data=decoding_result, ax=ax1)
ax1.set(xlim=[-41, 41], ylim=[-40, 40], xticks=np.arange(-40, 41, 20),
        xlabel='Stim. improvement ON blocks', ylabel='Stim. improvement OFF blocks')

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'LDA', f'lda_opto_improvement_probe_trials_scatter.png'))
