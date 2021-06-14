#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
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

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT')
save_path = join(save_path, '5HT')

# Load in results
if HISTOLOGY:
    stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons.csv'))
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'region', 'probe'])
else:
    stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons_no_histology.csv'))
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_no_histology.csv'))
    all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'probe'])

# Add genotype
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get max of left or right stim modulation
all_neurons['roc_light'] = all_neurons[['roc_l_light', 'roc_r_light']].values.max(1)
all_neurons['roc_stim'] = all_neurons[['roc_l_stim', 'roc_r_stim']].values.max(1)

# %% Plot
figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), dpi=150)
ax1.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax1.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons.loc[all_neurons['sert-cre'] == 1, 'roc_auc'],
            all_neurons.loc[all_neurons['sert-cre'] == 1, 'roc_stim'])
ax1.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus modulation', title='SERT')

ax2.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax2.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[all_neurons['sert-cre'] == 1, 'roc_auc'],
            all_neurons.loc[all_neurons['sert-cre'] == 1, 'roc_light'])
ax2.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus evoked 5-HT modulation')

ax3.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax3.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax3.scatter(all_neurons.loc[all_neurons['sert-cre'] == 0, 'roc_auc'],
            all_neurons.loc[all_neurons['sert-cre'] == 0, 'roc_stim'])
ax3.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus modulation', title='WT')

ax4.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax4.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax4.scatter(all_neurons.loc[all_neurons['sert-cre'] == 0, 'roc_auc'],
            all_neurons.loc[all_neurons['sert-cre'] == 0, 'roc_light'])
ax4.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus evoked 5-HT modulation')

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_stim_mod_summary'))


