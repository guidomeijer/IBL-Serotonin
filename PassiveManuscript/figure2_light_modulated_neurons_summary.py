#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import pearsonr
from serotonin_functions import paths, figure_style, load_subjects

# Settings
N_BINS = 30
MIN_NEURONS = 10
AP = [2, -1.5, -3.5]

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure2')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
expression_df = pd.read_csv(join(save_path, 'expression_levels.csv'))

# Add genotype and subject number
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'subject_nr'] = subjects.loc[subjects['subject'] == nickname].index[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]
wt_neurons = all_neurons[all_neurons['sert-cre'] == 0]

# Calculate percentage modulated neurons
all_mice = ((sert_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
             / sert_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
all_mice['sert-cre'] = 1
wt_mice = ((wt_neurons.groupby(['subject', 'subject_nr']).sum()['modulated']
            / wt_neurons.groupby(['subject', 'subject_nr']).size() * 100).to_frame().reset_index())
wt_mice['sert-cre'] = 0
all_mice = pd.concat((all_mice, wt_mice), ignore_index=True)
all_mice = all_mice.rename({0: 'perc_mod'}, axis=1)

# Merge dataframes
merged_df = pd.merge(all_mice, expression_df, on=['subject', 'sert-cre'])
merged_df = merged_df[merged_df['sert-cre'] == 1]

# %% Plot percentage mod neurons
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1), dpi=dpi)

f.subplots_adjust(bottom=0.2, left=0.35, right=0.85, top=0.9)
#sns.stripplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=3,
#              palette=[colors['sert'], colors['wt']], ax=ax1, jitter=0.2)

sns.swarmplot(x='sert-cre', y='perc_mod', data=all_mice, order=[1, 0], size=2.5,
              palette=[colors['sert'], colors['wt']], ax=ax1)
ax1.set(xticklabels=['SERT', 'WT'], ylabel='Mod. neurons (%)', ylim=[-1, 50], xlabel='',
        yticks=[0, 25, 50])

sns.despine(trim=True)
#plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_summary.pdf'))
plt.savefig(join(fig_path, 'light_mod_summary.jpg'), dpi=600)

# %% Plot percentage mod neurons vs expression
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.1), dpi=dpi)

f.subplots_adjust(bottom=0.3, left=0.32, right=0.88, top=0.9)
(
 so.Plot(merged_df, x='perc_mod', y='rel_fluo')
 .add(so.Dot(pointsize=2), color='subject_nr')
 .add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
 .scale(color='tab20')
 .on(ax1)
 .plot()
 )
ax1.set(xlim=[0, 50], xticks=[0, 25, 50],
        yticks=[0, 175, 350])
ax1.tick_params(axis='x', which='major', pad=2)
ax1.set_ylabel('Rel. expression (%)', rotation=90, labelpad=2)
ax1.set_xlabel('Mod. neurons (%)', rotation=0, labelpad=2)
r, p = pearsonr(merged_df['rel_fluo'], merged_df['perc_mod'])
print(f'correlation p-value: {p:.3f}')
ax1.text(25, 300, '**', fontsize=10)
sns.despine(trim=True)
plt.tight_layout()

plt.savefig(join(fig_path, 'light_mod_vs_expression.pdf'))
plt.savefig(join(fig_path, 'light_mod_vs_expression.jpg'), dpi=600)





