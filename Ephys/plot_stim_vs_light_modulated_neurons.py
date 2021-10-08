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
from scipy.stats import pearsonr
from os.path import join
from serotonin_functions import paths, figure_style, get_full_region_name, load_subjects

# Settings
N_BINS = 50
ARTIFACT_CUTOFF = 0.6
AXIS_LIM = 0.8

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'stim_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
light_neurons = light_neurons[light_neurons['roc_auc'] < ARTIFACT_CUTOFF]
all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'region', 'probe'])

# Add expression
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & all_neurons['task_responsive'], 'stim_mod_roc'],
            color=colors['sert'], s=6)
ax1.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~all_neurons['task_responsive'], 'stim_mod_roc'],
            color=colors['no-modulation'], s=6)
ax1.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Task modulation index', ylabel='5-HT stim modulation index', title='SERT')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & all_neurons['task_responsive'], 'stim_mod_roc'],
            color=colors['sert'], s=6)
ax2.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~all_neurons['task_responsive'], 'stim_mod_roc'],
            color=colors['no-modulation'], s=6)
ax2.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Task modulation index', ylabel='5-HT stim modulation index', title='WT')

ax3.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax3.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax3.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~(all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~(all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'],
            color=colors['no-modulation'], s=6)
ax3.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'],
            color=colors['sert'], s=6)
r, p = pearsonr(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
                all_neurons.loc[(all_neurons['sert-cre'] == 1)  & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'])
m, b = np.polyfit(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
                  all_neurons.loc[(all_neurons['sert-cre'] == 1)  & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'], 1)
ax3.plot([-1, 1], m*np.array([-1, 1]) + b, color=colors['sert'])
ax3.text(0.3, 0.6, f'r = {r:.2f}')
ax3.set(ylim=[-AXIS_LIM, AXIS_LIM], xlim=[-AXIS_LIM, AXIS_LIM], xlabel='Spontaneous 5-HT modulation',
        ylabel='Task-evoked 5-HT modulation', title='SERT', xticks=np.arange(-AXIS_LIM, AXIS_LIM+0.1, 0.4),
        yticks=np.arange(-AXIS_LIM, AXIS_LIM+0.1, 0.4))

ax4.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax4.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax4.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~(all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~(all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'],
            color=colors['no-modulation'], s=6)
ax4.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'roc_auc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['stim_modulated'] | (all_neurons['modulated'])), 'stim_mod_roc'],
            color=colors['sert'], s=6)
ax4.set(ylim=[-AXIS_LIM, AXIS_LIM], xlim=[-AXIS_LIM, AXIS_LIM], xlabel='Spontaneous 5-HT modulation',
        ylabel='Task-evoked 5-HT modulation', title='SERT')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.png'))
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.pdf'))