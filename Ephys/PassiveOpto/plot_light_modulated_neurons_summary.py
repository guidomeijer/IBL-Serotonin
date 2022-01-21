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
from serotonin_functions import paths, figure_style, load_subjects

# Settings
HISTOLOGY = True
N_BINS = 30
ARTIFACT_ROC = 0.6

# Paths
_, fig_path, save_path = paths()

# Load in results
if HISTOLOGY:
    all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
else:
    all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_no_histology.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Exclude artifact neurons
all_neurons = all_neurons[all_neurons['roc_auc'] < ARTIFACT_ROC]

# testing
#all_neurons = all_neurons.loc[all_neurons['subject'] == 'ZFM-02183']

# %%
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3.5), dpi=dpi)

ax1.hist(all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 0), 'roc_auc'],
         10, density=False, histtype='bar', color=colors['wt'])
ax1.hist([all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['enhanced'] == 1), 'roc_auc'],
          all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['suppressed'] == 1), 'roc_auc']],
         N_BINS, density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax1.set(xlim=[-.6, .6], xlabel='Modulation index', ylabel='Neuron count', title='SERT', ylim=[10**-1, 10**3],
        xticks=np.arange(-0.6, 0.61, 0.3), yscale='log', yticks=[10**-1, 10**0, 10**1, 10**2, 10**3],
        yticklabels=[0, 1, 10, 100, 1000])


ax2.hist([all_neurons.loc[(all_neurons['expression'] == 0) & (all_neurons['enhanced'] == 1), 'roc_auc'],
          all_neurons.loc[(all_neurons['expression'] == 0) & (all_neurons['suppressed'] == 1), 'roc_auc']],
         int(N_BINS/2), density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax2.set(xlim=[-0.6, 0.6], xlabel='Modulation index', ylabel='Neuron count', title='WT', ylim=[0, 5],
        xticks=np.arange(-0.6, 0.61, 0.3))

summary_df = all_neurons.groupby('subject').sum()
summary_df['n_neurons'] = all_neurons.groupby('subject').size()
summary_df['perc_mod'] = (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['expression'] = (summary_df['expression'] > 0).astype(int)

sns.swarmplot(x='expression', y='perc_mod', data=summary_df, ax=ax3,
              palette=[colors['wt'], colors['sert']])
ax3.set(ylabel='Modulated neurons (%)', xlabel='', xticklabels=['Wild type\ncontrol', 'Sert-Cre'], ylim=[0, 60])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.pdf'))
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.png'))

