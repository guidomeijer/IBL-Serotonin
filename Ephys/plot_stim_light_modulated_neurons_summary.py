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

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT')
save_path = join(save_path, '5HT')

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'region', 'probe'])

# Get max of left or right stim modulation
all_neurons['roc_light'] = all_neurons[['roc_l_light', 'roc_r_light']].values.max(1)
all_neurons['roc_stim'] = all_neurons[['roc_l_stim', 'roc_r_stim']].values.max(1)

# %% Plot
figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
ax1.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax1.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons['roc_auc'], all_neurons['roc_stim'])
ax1.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus modulation')

ax2.plot([0.5, 0.5], [0, 1], color=[.5, .5, .5], ls='--')
ax2.plot([0, 1], [.5, .5], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons['roc_auc'], all_neurons['roc_light'])
ax2.set(ylim=[0, 1], xlim=[0, 1], xlabel='5-HT modulation', ylabel='Stimulus evoked 5-HT modulation')

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_stim_mod_summary'))


