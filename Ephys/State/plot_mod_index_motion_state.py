#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:24:39 2022
By: Guido Meijer
"""
import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style

fig_path, data_path = paths()

# Load data
state_mod_neurons = pd.read_csv(join(data_path, 'motion_state_mod.csv'))
opto_mod_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(state_mod_neurons, opto_mod_neurons, on=['eid', 'pid', 'subject', 'date',
                                                                'neuron_id', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]


all_neurons = all_neurons[(all_neurons['sert-cre'] == 1) & (all_neurons['modulated'])]

#all_neurons = all_neurons[all_neurons['mod_index_late'] < 0]

all_neurons['abs_mod_inactive'] = all_neurons['mod_index_inactive'].abs()
all_neurons['abs_mod_active'] = all_neurons['mod_index_active'].abs()
all_neurons['mod_diff'] = all_neurons['mod_index_active'] - all_neurons['mod_index_inactive']

per_animal_df = all_neurons[all_neurons['modulated'] == True].groupby('subject').median(numeric_only=True)
all_df = all_neurons.groupby('subject').mean(numeric_only=True)
enh_df = all_neurons[all_neurons['mod_index_late'] > 0].groupby('subject').mean(numeric_only=True)
supp_df = all_neurons[all_neurons['mod_index_late'] < 0].groupby('subject').mean(numeric_only=True)

# %%
colors, dpi = figure_style()
sert_colors = [colors['wt'], colors['sert']]
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)

for i in per_animal_df.index:
    ax1.plot([0, 1], [enh_df.loc[i, 'mod_index_inactive'], enh_df.loc[i, 'mod_index_active']],
             '-o', markersize=2, color=colors['enhanced'])
_, p = wilcoxon(enh_df['mod_index_inactive'], enh_df['mod_index_active'])
ax1.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        title=f'Enhanced neurons (p={p:.2f})', ylim=[-0.12, 0.41])

for i in per_animal_df.index:
    ax2.plot([0, 1], [supp_df.loc[i, 'mod_index_inactive'], supp_df.loc[i, 'mod_index_active']],
             '-o', color=colors['suppressed'])
_, p = wilcoxon(supp_df['mod_index_inactive'], supp_df['mod_index_active'])
ax2.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        title=f'Suppressed neurons (p={p:.2f})', ylim=[-0.5, 0.01])

ax3.plot([-0.5, 1], [0, 0], color='grey', ls='--')
sns.swarmplot(data=all_df, y='mod_diff', ax=ax3, color=colors['sert'])
ax3.set(ylabel='Modulation change (up-down)', yticks=[-0.4, -0.3, -0.2, -0.1, 0, 0.1])

sns.despine(trim=True)
plt.tight_layout()

