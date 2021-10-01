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
from serotonin_functions import paths, figure_style, get_full_region_name, load_subjects

# Settings
HISTOLOGY = True
N_BINS = 50

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
if HISTOLOGY:
    stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons.csv'))
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'region', 'probe'])
else:
    stim_neurons = pd.read_csv(join(save_path, 'stim_light_modulated_neurons_no_histology.csv'))
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_no_histology.csv'))
    all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'cluster_id', 'eid', 'probe'])

# Add expression
subjects = load_subjects()
all_neurons = all_neurons[all_neurons['subject'].isin(subjects['subject'])]

# Exclude sert-cre mice without expression
all_neurons = all_neurons[~((all_neurons['expression'] == 0) & (all_neurons['sert-cre'] == 1))]

# Get max of left or right stim modulation
all_neurons['roc_light'] = all_neurons[['roc_l_light', 'roc_r_light']].values[
    np.arange(all_neurons.shape[0]), np.argmax(np.abs(all_neurons[['roc_l_light', 'roc_r_light']].values), axis=1)]
all_neurons['roc_stim'] = all_neurons[['roc_l_stim', 'roc_r_stim']].values[
    np.arange(all_neurons.shape[0]), np.argmax(np.abs(all_neurons[['roc_l_stim', 'roc_r_stim']].values), axis=1)]

# Get left or right modulated
all_neurons['mod_stim'] = all_neurons['mod_l_stim'] | all_neurons['mod_r_stim']
all_neurons['mod_light'] = all_neurons['mod_l_light'] | all_neurons['mod_r_light']

# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_stim'],
            color=colors['stim-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_stim'],
            color=colors['light-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_stim'],
            color=colors['both-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_stim'],
            color=colors['no-modulation'])
ax1.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='Stimulus evoked response', title='SERT')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_light'],
            color=colors['stim-significant'], label='Only stim')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_light'] & all_neurons['modulated'], 'roc_light'],
            color=colors['light-significant'], label='Only spontaneous')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_light'] & all_neurons['modulated'], 'roc_light'],
            color=colors['both-significant'], label='Both significant')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_light'],
            color=colors['no-modulation'], label='None')
ax2.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='Stimulus evoked 5-HT modulation')
ax2.legend(frameon=False, prop={'size': 20}, markerscale=2, loc='center left', bbox_to_anchor=(1, .5))

ax3.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax3.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_stim'],
            color=colors['stim-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_stim'],
            color=colors['light-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_stim'],
            color=colors['both-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_stim'],
            color=colors['no-modulation'])
ax3.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='Stimulus evoked response', title='WT')

ax4.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax4.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_light'],
            color=colors['stim-significant'], label='Only stim')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_light'] & all_neurons['modulated'], 'roc_light'],
            color=colors['light-significant'], label='Only spontaneous')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_light'] & all_neurons['modulated'], 'roc_light'],
            color=colors['both-significant'], label='Both significant')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_light'] & ~all_neurons['modulated'], 'roc_light'],
            color=colors['no-modulation'], label='None')
ax4.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='Stimulus evoked 5-HT modulation')
ax4.legend(frameon=False, prop={'size': 20}, markerscale=2, loc='center left', bbox_to_anchor=(1, .5))

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_stim_mod_summary'))

# %% Plot
colors = figure_style(return_colors=True)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_stim'] & ~all_neurons['modulated'], 'roc_0_stim'],
            color=colors['stim-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_stim'] & all_neurons['modulated'], 'roc_0_stim'],
            color=colors['light-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_stim'] & all_neurons['modulated'], 'roc_0_stim'],
            color=colors['both-significant'])
ax1.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_stim'] & ~all_neurons['modulated'], 'roc_0_stim'],
            color=colors['no-modulation'])
ax1.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='0% contrast evoked response', title='SERT')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_0_light'],
            color=colors['stim-significant'], label='Only stim')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_0_light'],
            color=colors['light-significant'], label='Only spontaneous')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_0_light'],
            color=colors['both-significant'], label='Both significant')
ax2.scatter(all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 1) & ~all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_0_light'],
            color=colors['no-modulation'], label='None')
ax2.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='0% contrast evoked 5-HT modulation')
ax2.legend(frameon=False, prop={'size': 20}, markerscale=2, loc='center left', bbox_to_anchor=(1, .5))

ax3.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax3.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_0_stim'],
            color=colors['stim-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_0_stim'],
            color=colors['light-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_stim'] & all_neurons['modulated'], 'roc_0_stim'],
            color=colors['both-significant'])
ax3.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_stim'] & ~all_neurons['modulated'], 'roc_0_stim'],
            color=colors['no-modulation'])
ax3.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='0% contrast evoked response', title='WT')

ax4.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax4.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_0_light'],
            color=colors['stim-significant'], label='Only stim')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_0_light'],
            color=colors['light-significant'], label='Only spontaneous')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & all_neurons['mod_0_light'] & all_neurons['modulated'], 'roc_0_light'],
            color=colors['both-significant'], label='Both significant')
ax4.scatter(all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_auc'],
            all_neurons.loc[(all_neurons['expression'] == 0) & ~all_neurons['mod_0_light'] & ~all_neurons['modulated'], 'roc_0_light'],
            color=colors['no-modulation'], label='None')
ax4.set(ylim=[-1.1, 1.1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation', ylabel='0% contrast evoked 5-HT modulation')
ax4.legend(frameon=False, prop={'size': 20}, markerscale=2, loc='center left', bbox_to_anchor=(1, .5))

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'light_0_stim_mod_summary'))

