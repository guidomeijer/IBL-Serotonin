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

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'neuron_id', 'eid', 'region', 'probe'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Get percentages of neurons
sert_neurons['choice_mod'] = sert_neurons['choice_p'] < 0.05
grouped_df = pd.DataFrame()
grouped_df['prior_all'] = (sert_neurons.groupby('subject').sum()['prior_modulated']
                           / sert_neurons.groupby('subject').size())
grouped_df['prior_opto'] = (sert_neurons[sert_neurons['modulated']].groupby('subject').sum()['prior_modulated']
                            / sert_neurons[sert_neurons['modulated']].groupby('subject').size())
grouped_df['prior_no_opto'] = (sert_neurons[~sert_neurons['modulated']].groupby('subject').sum()['prior_modulated']
                               / sert_neurons[~sert_neurons['modulated']].groupby('subject').size())

grouped_df['choice_all'] = (sert_neurons.groupby('subject').sum()['choice_mod']
                            / sert_neurons.groupby('subject').size())
grouped_df['choice_opto'] = (sert_neurons[sert_neurons['modulated']].groupby('subject').sum()['choice_mod']
                            / sert_neurons[sert_neurons['modulated']].groupby('subject').size())
grouped_df['choice_no_opto'] = (sert_neurons[~sert_neurons['modulated']].groupby('subject').sum()['choice_mod']
                                / sert_neurons[~sert_neurons['modulated']].groupby('subject').size())

# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax1.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & all_neurons['task_responsive'], 'opto_mod_roc'],
            color=colors['sert'], s=6)
ax1.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~all_neurons['task_responsive'], 'opto_mod_roc'],
            color=colors['grey'], s=6)
ax1.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Task modulation index', ylabel='5-HT stim modulation index', title='SERT')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & all_neurons['task_responsive'], 'opto_mod_roc'],
            color=colors['sert'], s=6)
ax2.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~all_neurons['task_responsive'], 'task_roc'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~all_neurons['task_responsive'], 'opto_mod_roc'],
            color=colors['grey'], s=6)
ax2.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Task modulation index', ylabel='5-HT stim modulation index', title='WT')

ax3.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax3.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax3.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~(all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & ~(all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'],
            color=colors['grey'], s=6)
ax3.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
            all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'],
            color=colors['sert'], s=6)
r, p = pearsonr(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
                all_neurons.loc[(all_neurons['sert-cre'] == 1)  & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'])
m, b = np.polyfit(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
                  all_neurons.loc[(all_neurons['sert-cre'] == 1)  & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'], 1)
ax3.plot([-1, 1], m*np.array([-1, 1]) + b, color=colors['sert'])
ax3.text(0.3, 0.6, f'r = {r:.2f}')
ax3.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation',
        ylabel='Task-evoked 5-HT modulation', title='SERT')

ax4.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax4.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax4.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~(all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & ~(all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'],
            color=colors['grey'], s=6)
ax4.scatter(all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'mod_index_late'],
            all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['opto_modulated'] | (all_neurons['modulated'])), 'opto_mod_roc'],
            color=colors['sert'], s=6)
ax4.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='Spontaneous 5-HT modulation',
        ylabel='Task-evoked 5-HT modulation', title='WT')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.png'))
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.pdf'))

# %%

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')

ax1.scatter(sert_neurons.loc[sert_neurons['modulated'], 'mod_index_late'],
            sert_neurons.loc[sert_neurons['modulated'], 'prior_roc'],
            color=colors['sert'], s=6)
ax1.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='5-HT modulation', ylabel='Prior modulation')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(sert_neurons.loc[sert_neurons['modulated'], 'mod_index_late'],
            sert_neurons.loc[sert_neurons['modulated'], 'choice_roc'],
            color=colors['sert'], s=6)
ax2.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='5-HT modulation', ylabel='Choice modulation')

sert_neurons['choice_roc_abs'] = sert_neurons['choice_roc'].abs()
sns.boxplot(x='modulated', y='choice_roc_abs', data=sert_neurons, ax=ax3, fliersize=0)
ax3.set(ylim=[0, 0.4])

sert_neurons['prior_roc_abs'] = sert_neurons['prior_roc'].abs()
sns.boxplot(x='modulated', y='prior_roc_abs', data=sert_neurons, ax=ax4, fliersize=0)


