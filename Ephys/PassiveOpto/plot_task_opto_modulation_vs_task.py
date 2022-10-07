#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from os.path import join
from serotonin_functions import (paths, figure_style, get_full_region_name, load_subjects,
                                 high_level_regions)

# Settings
N_BINS = 50

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
stim_neurons = pd.read_csv(join(save_path, 'task_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'neuron_id', 'eid', 'region', 'probe', 'pid'])
all_neurons['high_level_region'] = high_level_regions(all_neurons['region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Calculate adjusted modulation index
sert_neurons['mod_adjusted'] = sert_neurons['opto_mod_roc'] - sert_neurons['mod_index_late']

# Get for which type neurons are modulated
sert_neurons.loc[sert_neurons['opto_modulated'] & sert_neurons['modulated'], 'sig_modulation'] = 'Both'
sert_neurons.loc[~sert_neurons['opto_modulated'] & sert_neurons['modulated'], 'sig_modulation'] = 'Passive'
sert_neurons.loc[sert_neurons['opto_modulated'] & ~sert_neurons['modulated'], 'sig_modulation'] = 'Task'

# Get percentages of neurons
all_neurons['choice_mod'] = all_neurons['choice_no_stim_p'] < 0.05
#all_neurons = all_neurons[all_neurons['high_level_region'] == 'Frontal']
grouped_df = pd.DataFrame()
grouped_df['prior_all'] = (all_neurons.groupby('subject').sum()['prior_modulated']
                           / all_neurons.groupby('subject').size())
grouped_df['prior_opto'] = (all_neurons[all_neurons['modulated']].groupby('subject').sum()['prior_modulated']
                            / all_neurons[all_neurons['modulated']].groupby('subject').size())
grouped_df['prior_no_opto'] = (all_neurons[~all_neurons['modulated']].groupby('subject').sum()['prior_modulated']
                               / all_neurons[~all_neurons['modulated']].groupby('subject').size())

grouped_df['choice_all'] = (all_neurons.groupby('subject').sum()['choice_mod']
                            / all_neurons.groupby('subject').size())
grouped_df['choice_opto'] = (all_neurons[all_neurons['modulated']].groupby('subject').sum()['choice_mod']
                            / all_neurons[all_neurons['modulated']].groupby('subject').size())
grouped_df['choice_no_opto'] = (all_neurons[~all_neurons['modulated']].groupby('subject').sum()['choice_mod']
                                / all_neurons[~all_neurons['modulated']].groupby('subject').size())
grouped_df['spont_perc'] = (all_neurons.groupby('subject').sum()['modulated']
                            / all_neurons.groupby('subject').size()) * 100
grouped_df['task_perc'] = (all_neurons.groupby('subject').sum()['opto_modulated']
                           / all_neurons.groupby('subject').size()) * 100

all_neurons['task_roc_abs'] = all_neurons['task_roc'].abs()
grouped_df['mod_choice_roc'] = all_neurons[all_neurons['modulated']].groupby('subject').mean()['task_roc_abs']
grouped_df['no_mod_choice_roc'] = all_neurons[~all_neurons['modulated']].groupby('subject').mean()['task_roc_abs']

grouped_df['choice_stim_roc'] = all_neurons.groupby('subject').mean()['choice_stim_roc']
grouped_df['choice_no_stim_roc'] = all_neurons.groupby('subject').mean()['choice_no_stim_roc']

all_neurons['choice_diff'] = all_neurons['choice_no_stim_roc'] - all_neurons['choice_stim_roc']
grouped_df['choice_diff'] = all_neurons.groupby('subject').mean()['choice_diff']

all_neurons['choice_roc_abs'] = all_neurons['choice_no_stim_roc'].abs()
grouped_df['choice_mod_roc'] = all_neurons[all_neurons['modulated']].groupby('subject').mean()['choice_roc_abs']
grouped_df['choice_no_mod_roc'] = all_neurons[~all_neurons['modulated']].groupby('subject').mean()['choice_roc_abs']
all_neurons['mod_index_abs'] = all_neurons['mod_index_late'].abs()
all_neurons['opto_mod_abs'] = all_neurons['opto_mod_roc'].abs()
grouped_df['spont_opto_mod'] = all_neurons[all_neurons['modulated'] | all_neurons['opto_modulated']].groupby('subject').mean()['mod_index_abs']
grouped_df['task_opto_mod'] = all_neurons[all_neurons['modulated'] | all_neurons['opto_modulated']].groupby('subject').mean()['opto_mod_abs']

all_neurons['choice_stim_roc_abs'] = all_neurons['choice_stim_roc'].abs()
all_neurons['choice_no_stim_roc_abs'] = all_neurons['choice_no_stim_roc'].abs()
grouped_df['choice_stim_roc'] = all_neurons.groupby('subject').mean()['choice_stim_roc']
grouped_df['choice_no_stim_roc'] = all_neurons.groupby('subject').mean()['choice_no_stim_roc']
grouped_df['choice_stim_roc_abs'] = all_neurons.groupby('subject').mean()['choice_stim_roc_abs']
grouped_df['choice_no_stim_roc_abs'] = all_neurons.groupby('subject').mean()['choice_no_stim_roc_abs']

grouped_df = grouped_df.reset_index()
for i, nickname in enumerate(np.unique(grouped_df['subject'])):
    grouped_df.loc[grouped_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]


# %% Plot
colors, dpi = figure_style()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.5, 3.5), gridspec_kw={'width_ratios':[1.5,1]}, dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--', zorder=0)
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--', zorder=0)
mod_neurons = sert_neurons[sert_neurons['opto_modulated'] | sert_neurons['modulated']]
(
     so.Plot(mod_neurons, x='task_roc', y='mod_index_late')
     .add(so.Dot(pointsize=2), color='sig_modulation')
     .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .limit(x=[-1, 1], y=[-1, 1])
     .label(x='Task modulation', y='Spontaneous 5-TH modulation')
     .on(ax1)
     .plot()
)
r, p = pearsonr(mod_neurons['task_roc'], mod_neurons['mod_index_late'])
ax1.text(0.4, -0.9, f'r = {r:.2f}', fontsize=7)

for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax2.plot([1, 2], [grouped_df.loc[i, 'choice_stim_roc'], grouped_df.loc[i, 'choice_no_stim_roc']], '-o',
             color=colors['sert'], markersize=2)
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax2.plot([1, 2], [grouped_df.loc[i, 'choice_stim_roc'], grouped_df.loc[i, 'choice_no_stim_roc']], '-o',
             color=colors['wt'], markersize=2)
ax2.set(ylabel='Modulation index', xticks=[1, 2], xticklabels=['Stim', 'No-stim'])

mod_neurons['choice_diff'] = mod_neurons['choice_no_stim_roc'] - mod_neurons['choice_stim_roc']
sns.boxplot(y='choice_diff', data=mod_neurons, ax=ax3)

for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'mod_choice_roc'], grouped_df.loc[i, 'no_mod_choice_roc']], '-o',
             color=colors['sert'], markersize=2)
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'mod_choice_roc'], grouped_df.loc[i, 'no_mod_choice_roc']], '-o',
             color=colors['wt'], markersize=2)
ax4.set(ylabel='Modulation index', xticks=[1, 2], xticklabels=['Stim', 'No-stim'])

sns.despine(trim=True)
plt.tight_layout()


