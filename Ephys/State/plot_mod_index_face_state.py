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
state_mod_neurons = pd.read_csv(join(data_path, 'state_modulation.csv'))
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

per_animal_df = all_neurons[all_neurons['modulated'] == True].groupby('subject').median(numeric_only=True)
enh_df = all_neurons[all_neurons['mod_index_late'] > 0].groupby('subject').mean(numeric_only=True)
supp_df = all_neurons[all_neurons['mod_index_late'] < 0].groupby('subject').mean(numeric_only=True)

# %%
colors, dpi = figure_style()
sert_colors = [colors['wt'], colors['sert']]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

for i in per_animal_df.index:
    ax1.plot([0, 1], [enh_df.loc[i, 'mod_index_inactive'], enh_df.loc[i, 'mod_index_active']],
             '-o', markersize=2, color=colors['enhanced'])
_, p = wilcoxon(enh_df['mod_index_inactive'], enh_df['mod_index_active'])
ax1.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        title='Enhanced neurons', ylim=[-0.1, 0.4])

for i in per_animal_df.index:
    ax2.plot([0, 1], [supp_df.loc[i, 'mod_index_inactive'], supp_df.loc[i, 'mod_index_active']],
             '-o', color=colors['suppressed'])
_, p = wilcoxon(supp_df['mod_index_inactive'], supp_df['mod_index_active'])
ax2.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Inactive', 'Active'], xlabel='State',
        title='Suppressed neurons', ylim=[-0.5, 0.1])

sns.despine(trim=True)
plt.tight_layout()