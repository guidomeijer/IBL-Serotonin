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
MIN_NEURONS = 10
AP = [2, -1.5, -3.5]

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]
wt_neurons = all_neurons[all_neurons['sert-cre'] == 0]

all_mice = (sert_neurons.groupby('subject').sum()['modulated'] / sert_neurons.groupby('subject').size() * 100).to_frame()
all_mice['sert-cre'] = 1
wt_mice = (wt_neurons.groupby('subject').sum()['modulated'] / wt_neurons.groupby('subject').size() * 100).to_frame()
wt_mice['sert-cre'] = 0
all_mice = pd.concat((all_mice, wt_mice), ignore_index=True)
all_mice = all_mice.rename({0: 'perc_mod'}, axis=1)

# %%

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)

ax1.hist(sert_neurons.loc[sert_neurons['modulated'] == 1, 'mod_index_early'], bins=N_BINS,
         histtype='step', color=colors['early'], label='Early')
ax1.hist(sert_neurons.loc[sert_neurons['modulated'] == 1, 'mod_index_late'], bins=N_BINS,
         histtype='step', color=colors['late'], label='Late')
ax1.legend(frameon=False)
ax1.set(ylabel='Neuron count', xlabel='Modulation index', ylim=[0, 200])

sns.swarmplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0],
              palette=[colors['sert'], colors['wt']], ax=ax2)
ax2.set(xticklabels=['SERT', 'WT'], ylabel='5-HT modulated neurons (%)', ylim=[0, 80], xlabel='')

sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_summary.pdf'))
plt.savefig(join(fig_path, 'light_mod_summary.jpg'), dpi=300)
