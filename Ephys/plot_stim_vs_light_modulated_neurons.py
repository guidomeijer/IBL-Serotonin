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
all_neurons = pd.merge(stim_neurons, light_neurons, on=['subject', 'date', 'neuron_id', 'eid', 'region', 'probe'])
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

all_neurons['choice_roc_abs'] = all_neurons['choice_no_stim_roc'].abs()
grouped_df['choice_mod_roc'] = all_neurons[all_neurons['modulated']].groupby('subject').mean()['choice_roc_abs']
grouped_df['choice_no_mod_roc'] = all_neurons[~all_neurons['modulated']].groupby('subject').mean()['choice_roc_abs']
all_neurons['mod_index_abs'] = all_neurons['mod_index_late'].abs()
all_neurons['opto_mod_abs'] = all_neurons['opto_mod_roc'].abs()
grouped_df['spont_opto_mod'] = all_neurons[all_neurons['modulated'] | all_neurons['opto_modulated']].groupby('subject').mean()['mod_index_abs']
grouped_df['task_opto_mod'] = all_neurons[all_neurons['modulated'] | all_neurons['opto_modulated']].groupby('subject').mean()['opto_mod_abs']


grouped_df = grouped_df.reset_index()
for i, nickname in enumerate(np.unique(grouped_df['subject'])):
    grouped_df.loc[grouped_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]


# %% Plot
colors, dpi = figure_style()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.5, 3.5), gridspec_kw={'width_ratios':[1.5,1]}, dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--', zorder=0)
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--', zorder=0)
(
     so.Plot(sert_neurons, x='mod_index_late', y='opto_mod_roc', color='sig_modulation')
     .add(so.Dot(pointsize=2))
     .add(so.Line(color='k'), so.PolyFit(order=1), color=None)
     .limit(x=[-1, 1], y=[-1, 1])
     .label(x='Spontaneous 5-HT modulation', y='Task evoked 5-TH modulation')
     .on(ax1)
     .plot()
)
r, p = pearsonr(sert_neurons.loc[(sert_neurons['opto_modulated'] | (sert_neurons['modulated'])), 'mod_index_late'],
                sert_neurons.loc[(sert_neurons['opto_modulated'] | (sert_neurons['modulated'])), 'opto_mod_roc'])
ax1.text(0.3, -0.8, f'r = {r:.2f}', fontsize=7)
ax1.legend(frameon=False, prop={'size': 5}, loc='upper left')
legend = f.legends.pop(0)
ax1.legend(legend.legendHandles, [t.get_text() for t in legend.texts], frameon=False,
           prop={'size': 5}, loc='upper left')



for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax2.plot([1, 2], [grouped_df.loc[i, 'spont_opto_mod'], grouped_df.loc[i, 'task_opto_mod']], '-o',
             color=colors['sert'], markersize=2)
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax2.plot([1, 2], [grouped_df.loc[i, 'spont_opto_mod'], grouped_df.loc[i, 'task_opto_mod']], '-o',
             color=colors['wt'], markersize=2)
ax2.set(ylabel='Modulation index', xticks=[1, 2], xticklabels=['Spontaneous', 'Task'],
        yticks=[0, 0.1, 0.2, 0.3, 0.4])

for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'spont_perc'], grouped_df.loc[i, 'task_perc']], '-o',
             color=colors['sert'], markersize=2)
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'spont_perc'], grouped_df.loc[i, 'task_perc']], '-o',
             color=colors['wt'], markersize=2)
ax4.set(ylabel='Modulated neurons (%)', xticks=[1, 2], xticklabels=['Spontaneous', 'Task'],
        yticks=[0, 10, 20, 30, 40, 50, 60, 70])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.png'))
plt.savefig(join(fig_path, 'Ephys', 'stim_vs_light_modulation.pdf'))

# %%

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
ax1.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax1.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')

ax1.scatter(all_neurons.loc[all_neurons['modulated'], 'mod_index_late'],
            all_neurons.loc[all_neurons['modulated'], 'prior_roc'],
            color=colors['sert'], s=6)
ax1.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='5-HT modulation', ylabel='Prior modulation')

ax2.plot([0, 0], [-1, 1], color=[.5, .5, .5], ls='--')
ax2.plot([-1, 1], [0, 0], color=[.5, .5, .5], ls='--')
ax2.scatter(all_neurons.loc[all_neurons['modulated'], 'mod_index_late'],
            all_neurons.loc[all_neurons['modulated'], 'choice_no_stim_roc'],
            color=colors['sert'], s=6)
ax2.set(ylim=[-1, 1], xlim=[-1, 1], xlabel='5-HT modulation', ylabel='Choice modulation')

for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax3.plot([1, 2], [grouped_df.loc[i, 'choice_mod_roc'], grouped_df.loc[i, 'choice_no_mod_roc']],
             color=colors['sert'])
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax3.plot([1, 2], [grouped_df.loc[i, 'choice_mod_roc'], grouped_df.loc[i, 'choice_no_mod_roc']],
             color=colors['wt'])


for i in grouped_df[grouped_df['sert-cre'] == 1].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'prior_no_opto'], grouped_df.loc[i, 'prior_opto']],
             color=colors['sert'])
for i in grouped_df[grouped_df['sert-cre'] == 0].index:
    ax4.plot([1, 2], [grouped_df.loc[i, 'prior_no_opto'], grouped_df.loc[i, 'prior_opto']],
             color=colors['wt'])

