#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:12:46 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from os.path import join
from serotonin_functions import paths, figure_style, load_subjects

# Settings

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PassiveState')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
pupil_neurons = pd.read_csv(join(save_path, 'state_pupil_neurons.csv'))
lfp_neurons = pd.read_csv(join(save_path, 'state_lfp_neurons.csv'))

# Merge dfs
merged_pupil_df = pd.merge(all_neurons, pupil_neurons, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])
merged_lfp_df = pd.merge(all_neurons, lfp_neurons, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])

# Get positive or negative modulation
merged_lfp_df['mod_sign'] = np.sign(merged_lfp_df['mod_index_late'])

# Get absolute modulation
merged_pupil_df['mod_index_small_abs'] = merged_pupil_df['mod_index_small'].abs()
merged_pupil_df['mod_index_large_abs'] = merged_pupil_df['mod_index_large'].abs()

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    merged_pupil_df.loc[merged_pupil_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    merged_lfp_df.loc[merged_lfp_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get average per animal
pupil_df = merged_pupil_df.groupby(['subject']).median().reset_index()
lfp_df = merged_lfp_df.groupby(['subject', 'mod_sign', 'freq_band']).median().reset_index()

# Drop 0 modulation
lfp_df = lfp_df[lfp_df['mod_sign'] != 0]


# %% Plot
colors, dpi = figure_style()

f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
for i in pupil_df[pupil_df['sert-cre'] == 1].index:
    ax.plot([1, 2], [pupil_df.loc[i, 'mod_index_small_abs'], pupil_df.loc[i, 'mod_index_large_abs']],
            color=colors['sert'])
for i in pupil_df[pupil_df['sert-cre'] == 0].index:
    ax.plot([1, 2], [pupil_df.loc[i, 'mod_index_small_abs'], pupil_df.loc[i, 'mod_index_large_abs']],
            color=colors['wt'])
_, p = wilcoxon(pupil_df[pupil_df['sert-cre'] == 1]['mod_index_small_abs'],
                pupil_df[pupil_df['sert-cre'] == 1]['mod_index_large_abs'])
if p < 0.05:
    ax.text(1.5, 0.18, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
    print(f'Pupil: p = {p:.3f}')
ax.set(xticks=[1, 2], xticklabels=['Small pupil', 'Large pupil'], ylabel='Abs. modulation index',
       yticks=[0, .1, .2])

sns.despine(trim=True)
plt.tight_layout()

# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 1.75), dpi=dpi)
for i in lfp_df[lfp_df['freq_band'] == 'alpha'].index:
    if lfp_df.loc[i, 'sert-cre'] == 1:
        ax1.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['sert'])
    else:
        ax1.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['wt'])
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'alpha') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'alpha') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_high'])
if p < 0.05:
    print(f'Alpha enhancement: p = {p:.3f}')
    ax1.text(1.5, 0.16, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'alpha') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'alpha') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_high'])
if p < 0.05:
    print(f'Alpha suppression: p = {p:.3f}')
    ax1.text(1.5, -0.2, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
ax1.plot([0.8, 2.2], [0, 0], ls='--', color='grey')
ax1.set(xticks=[1, 2], xticklabels=['Low', 'High'], ylabel='Modulation index',
        yticks=[-0.2, -0.1, 0, 0.1, 0.2], title='Alpha (9-15 Hz)')
ax1.text(-0.2, -0.25, 'LFP Power:')


for i in lfp_df[lfp_df['freq_band'] == 'beta'].index:
    if lfp_df.loc[i, 'sert-cre'] == 1:
        ax2.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['sert'])
    else:
        ax2.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['wt'])
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'beta') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'beta') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_high'])
if p < 0.05:
    print(f'Beta enhancement: p = {p:.3f}')
    ax2.text(1.5, 0.16, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'beta') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'beta') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_high'])
if p < 0.05:
    print(f'Beta suppression: p = {p:.3f}')
    ax2.text(1.5, -0.2, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
ax2.plot([0.8, 2.2], [0, 0], ls='--', color='grey')
ax2.set(xticks=[1, 2], xticklabels=['Low', 'High'],
        yticks=[-0.2, -0.1, 0, .1, .2], title='Beta (15-30 Hz)')


for i in lfp_df[lfp_df['freq_band'] == 'gamma'].index:
    if lfp_df.loc[i, 'sert-cre'] == 1:
        ax3.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['sert'])
    else:
        ax3.plot([1, 2], [lfp_df.loc[i, 'mod_index_low'], lfp_df.loc[i, 'mod_index_high']],
                 color=colors['wt'])
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'gamma') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'gamma') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == 1), 'mod_index_high'])
if p < 0.05:
    print(f'Gamma enhancement: p = {p:.3f}')
    ax3.text(1.5, 0.16, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
_, p = wilcoxon(lfp_df.loc[(lfp_df['freq_band'] == 'gamma') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_low'],
                lfp_df.loc[(lfp_df['freq_band'] == 'gamma') & (lfp_df['sert-cre'] == 1) & (lfp_df['mod_sign'] == -1), 'mod_index_high'])
if p < 0.05:
    print(f'Gamma suppression: p = {p:.3f}')
    ax3.text(1.5, -0.2, '*', fontsize=12, fontweight='bold', color=colors['sert'], ha='center')
ax3.plot([0.8, 2.2], [0, 0], ls='--', color='grey')
ax3.set(xticks=[1, 2], xticklabels=['Low', 'High'],
        yticks=[-0.2, -0.1, 0, .1, .2], title='Gamma (30-100 Hz)')


sns.despine(trim=True)
plt.tight_layout()

