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
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style

fig_path, data_path = paths()

# Load data
state_mod_neurons = pd.read_csv(join(data_path, 'neural_state_modulation.csv'))
opto_mod_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(state_mod_neurons, opto_mod_neurons, on=['eid', 'pid', 'subject', 'date',
                                                                'neuron_id', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

enh_df = all_neurons[all_neurons['modulated'] & (all_neurons['mod_index_late'] > 0)].groupby('subject').mean()
supp_df = all_neurons[all_neurons['modulated'] & (all_neurons['mod_index_late'] < 0)].groupby('subject').mean()

# %%
colors, dpi = figure_style()
sert_colors = [colors['wt'], colors['sert']]

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

for i in enh_df.index:
    ax1.plot([0, 1], [enh_df.loc[i, 'mod_index_low'], enh_df.loc[i, 'mod_index_high']],
             '-o', color=sert_colors[enh_df.loc[i, 'sert-cre'].astype(int)], markersize=2)
ax1.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        yticks=[0, 0.1, 0.2, 0.3], title='Enhanced neurons')

for i in supp_df.index:
    ax2.plot([0, 1], [supp_df.loc[i, 'mod_index_low'], supp_df.loc[i, 'mod_index_high']],
             '-o', color=sert_colors[enh_df.loc[i, 'sert-cre'].astype(int)], markersize=2)
ax2.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        yticks=[-0.3, -0.2, -0.1, 0], title='Suppressed neurons')


sns.despine(trim=True)
plt.tight_layout()