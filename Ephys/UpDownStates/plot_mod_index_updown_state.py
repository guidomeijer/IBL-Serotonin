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
time_win = 'late'

# Load data
state_mod_neurons = pd.read_csv(join(data_path, 'updown_states_modulation.csv'))
state_mod_neurons = state_mod_neurons.rename(columns={'acronym': 'region'})
opto_mod_neurons = pd.read_csv(join(data_path, 'light_modulated_neurons.csv'))
all_neurons = pd.merge(state_mod_neurons, opto_mod_neurons, on=['eid', 'pid', 'subject', 'date',
                                                                'neuron_id', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

all_neurons = all_neurons[(all_neurons['modulated']) & (all_neurons['sert-cre'])]  # select modulated neurons in sert-cre animals
all_neurons['mod_down_abs'] = all_neurons[f'mod_down_{time_win}'].abs()
all_neurons['mod_up_abs'] = all_neurons[f'mod_up_{time_win}'].abs()
all_neurons['mod_diff'] = all_neurons[f'mod_up_{time_win}'] - all_neurons[f'mod_down_{time_win}']

all_df = all_neurons.groupby('subject').mean(numeric_only=True)
enh_df = all_neurons[all_neurons[f'mod_index_{time_win}'] > 0].groupby('subject').mean(numeric_only=True)
supp_df = all_neurons[all_neurons[f'mod_index_{time_win}'] < 0].groupby('subject').mean(numeric_only=True)


# %%
colors, dpi = figure_style()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 1.75), dpi=dpi)

for i in enh_df.index:
    ax1.plot([0, 1], [enh_df.loc[i, f'mod_down_{time_win}'], enh_df.loc[i, f'mod_up_{time_win}']],
             '-o', color=colors['enhanced'], markersize=2)
ax1.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State',
        yticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4], title='Enhanced neurons', xlim=[-0.2, 1.2], ylim=[-0.1, 0.4])

for i in supp_df.index:
    ax2.plot([0, 1], [supp_df.loc[i, f'mod_down_{time_win}'], supp_df.loc[i, f'mod_up_{time_win}']],
             '-o', color=colors['suppressed'], markersize=2)
ax2.set(ylabel='Modulation index', xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State',
        yticks=[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1], title='Suppressed neurons', xlim=[-0.2, 1.2])

ax3.plot([-0.5, 1], [0, 0], color='grey', ls='--')
sns.swarmplot(data=all_df, y='mod_diff', ax=ax3, color=colors['sert'])
ax3.set(ylabel='Modulation change (up-down)', yticks=[-0.4, -0.3, -0.2, -0.1, 0, 0.1])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'mod_index_updown_state.jpg'), dpi=600)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot([-1, 1], [-1, 1], color='grey', ls='--')
sns.scatterplot(data=all_neurons, x=f'mod_down_{time_win}', y=f'mod_up_{time_win}', ax=ax1)
ax1.set(xlabel='Modulation during down state', ylabel='Modulation during up state', title='Per neuron')

ax2.hist(all_neurons['mod_diff'], bins=20)
ax2.set(xlabel='Change in modulation (up-down)', ylabel='Neuron count', xlim=[-1, 1], yticks=[0, 75, 150])

plt.tight_layout()
sns.despine(trim=True)