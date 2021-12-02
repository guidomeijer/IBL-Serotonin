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
from scipy.stats import pearsonr
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from serotonin_functions import paths, get_full_region_name, figure_style

# Settings
_, fig_path, save_path = paths()
MIN_NEURONS = 5
PRE_TIME = 0
POST_TIME = 0.3
ARTIFACT_ROC = 0.6
TARGET = 'block'

# Load in decoding results
decoding_result = pd.read_csv(join(save_path, f'lda_decoding_{TARGET}_{PRE_TIME}_{POST_TIME}.csv'))
decoding_result['delta_block'] = (decoding_result['acc_block_on'] - decoding_result['acc_block_off']) * 100
decoding_result['full_region'] = get_full_region_name(decoding_result['region'])

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Calculate average decoding performance per region
summary_decoding = decoding_result.groupby('region').mean()['delta_block'].reset_index()

# Load in stimulus modulation results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['region']) if 'void' in j])

# Add expression
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(light_neurons['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
light_neurons = light_neurons[light_neurons['sert-cre'] == 1]

# Exclude artifact neurons
light_neurons = light_neurons[light_neurons['roc_auc'] < ARTIFACT_ROC]

# Calculate summary statistics
summary_df = light_neurons.groupby(['region', 'eid']).sum()
summary_df['n_neurons'] = light_neurons.groupby(['region', 'eid']).size()
summary_df = summary_df.reset_index()
summary_df['perc_enh'] =  (summary_df['enhanced'] / summary_df['n_neurons']) * 100
summary_df['perc_supp'] =  (summary_df['suppressed'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
summary_df['full_region'] = get_full_region_name(summary_df['region'])
summary_df['ratio'] = summary_df['perc_enh'] - summary_df['perc_supp']
summary_df['perc_supp'] = -summary_df['perc_supp']
summary_df['total_perc'] = summary_df['perc_supp'].abs() + summary_df['perc_enh']

summary_df = summary_df.groupby('region').mean().reset_index()

# Merge dataframes
merged_df = pd.merge(summary_df, summary_decoding, on='region')

# Perform correlation
r_ratio, p_ratio = pearsonr(merged_df['ratio'], merged_df['delta_block'])
r_total, p_total = pearsonr(merged_df['total_perc'], merged_df['delta_block'])

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
sns.regplot(x='ratio', y='delta_block', data=merged_df, ax=ax1)
ax1.set(xlim=[-25, 20], ylim=[-30, 30], xlabel='Modulation ratio', ylabel='Decoding improvement',
        title=f'r={r_ratio:.2f}, p={p_ratio:.2f}')

sns.regplot(x='total_perc', y='delta_block', data=merged_df, ax=ax2)
ax2.set(xlim=[-0.5, 30], ylim=[-40, 40], xlabel='Total abs. modulation', ylabel='Decoding improvement',
        title=f'r={r_total:.2f}, p={p_total:.2f}')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'lda_vs_modulation.pdf'))


