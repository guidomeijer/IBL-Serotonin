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
from serotonin_functions import paths, figure_style, get_full_region_name

# Settings
HISTOLOGY = False
N_BINS = 50

# Paths
_, fig_path, save_path = paths()

# Load in results
if HISTOLOGY:
    all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
else:
    all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_no_histology.csv'))

# Add genotype
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Drop ZFM-02180 for now
all_neurons = all_neurons.loc[all_neurons['subject'] != 'ZFM-02184']

# %%
colors = figure_style(return_colors=True)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), dpi=150)
ax1.hist([all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'] == 0), 'roc_auc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['enhanced'] == 1), 'roc_auc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['suppressed'] == 1), 'roc_auc']],
         N_BINS, density=True, histtype='bar', stacked=True,
         color=[colors['no-modulation'], colors['enhanced'], colors['suppressed']])
ax1.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count', title='SERT')

ax2.hist([all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['modulated'] == 0), 'roc_auc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['enhanced'] == 1), 'roc_auc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['suppressed'] == 1), 'roc_auc']],
         int(N_BINS/2), density=True, histtype='bar', stacked=True,
         color=[colors['no-modulation'], colors['enhanced'], colors['suppressed']])
ax2.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count', title='WT')

summary_df = all_neurons.groupby('subject').sum()
summary_df['n_neurons'] = all_neurons.groupby('subject').size()
summary_df['perc_mod'] = (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['sert-cre'] = (summary_df['sert-cre'] > 0).astype(int)

sns.swarmplot(x='sert-cre', y='perc_mod', data=summary_df, ax=ax3,
              palette=[colors['wt'], colors['sert']], s=10)
ax3.set(ylabel='Light modulated neurons (%)', xlabel='', xticklabels=['WT', 'SERT'], ylim=[0, 70])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'opto_modulation_summary'))


