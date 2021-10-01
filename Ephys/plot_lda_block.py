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
TARGET = 'stim'

# Load in results
decoding_result = pd.read_csv(join(save_path, f'lda_decoding_{TARGET}_{PRE_TIME}_{POST_TIME}.csv'))

# Calculate delta
decoding_result['delta_block'] = (decoding_result['acc_block_on'] - decoding_result['acc_block_off']) * 100

# Get full region names
decoding_result['full_region'] = get_full_region_name(decoding_result['region'])

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'mean'] = decoding_result.loc[decoding_result['region'] == region, 'delta_block'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(decoding_result['region'] == region)

# Apply selection
decoding_result = decoding_result[decoding_result['mean'].abs() >= MIN_IMPROVEMENT]

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

decoding_sert = decoding_result[decoding_result['sert-cre'] == 1]
sort_regions = decoding_sert.groupby('full_region').max().sort_values('mean', ascending=False).reset_index()['full_region']
sns.barplot(x='mean', y='full_region', data=decoding_sert, order=sort_regions,
            ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2", ax=ax1)
sns.swarmplot(x='delta_block', y='full_region', data=decoding_sert, order=sort_regions, ax=ax1)

ax1.set(xlabel='Stimulation induced decoding improvement of prior (% correct)', ylabel='', xlim=[-30, 30])

"""
decoding_wt = decoding_result[decoding_result['sert-cre'] == 0]
sort_regions = decoding_wt.groupby('full_region').max().sort_values('mean', ascending=False).reset_index()['full_region']
sns.barplot(x='mean', y='full_region', data=decoding_wt, order=sort_regions,
            ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2", ax=ax2)
sns.swarmplot(x='delta_block', y='full_region', data=decoding_wt, order=sort_regions, ax=ax1)

ax2.set(xlabel='Stimulation induced decoding improvement of prior (% correct)', ylabel='', xlim=[-25, 25])
"""

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'LDA', f'lda_opto_improvement_{TARGET}_{PRE_TIME}_{POST_TIME}.png'))
