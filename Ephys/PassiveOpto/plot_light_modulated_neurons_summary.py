#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, load_subjects

# Settings
HISTOLOGY = True
N_BINS = 30
MIN_NEURONS = 10

# Paths
fig_path, save_path = paths()
map_path = join(fig_path, 'Ephys', 'BrainMaps')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Get percentage modulated per region
reg_neurons = (sert_neurons.groupby('region').sum()['modulated'] / sert_neurons.groupby('region').size() * 100).to_frame()
reg_neurons = reg_neurons.rename({0: 'percentage'}, axis=1)
reg_neurons['mod_early'] = sert_neurons.groupby('region').median()['mod_index_early']
reg_neurons['mod_late'] = sert_neurons.groupby('region').median()['mod_index_late']
reg_neurons['n_neurons'] = sert_neurons.groupby(['region']).size()
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['region'] != 'root']


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)

#ax2.hist(all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 0), 'mod_index_late'],
#         10, density=False, histtype='bar', color=colors['wt'])
ax1.hist([all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 1), 'mod_index_late'],
          all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 1), 'mod_index_early']],
         N_BINS, density=False, histtype='step', stacked=False,
         color=[colors['late'], colors['early']])
ax1.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count',
        xticks=np.arange(-1, 1.1, 0.5), ylim=[0, 100], yticks=[0, 50, 100])
ax1.legend(['Early', 'Late'], frameon=False, bbox_to_anchor=(0.55, 0.7))

summary_df = all_neurons.groupby('subject').sum()
summary_df['n_neurons'] = all_neurons.groupby('subject').size()
summary_df['perc_mod'] = (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['expression'] = (summary_df['expression'] > 0).astype(int)

sns.swarmplot(x='expression', y='perc_mod', data=summary_df, ax=ax2,
              palette=[colors['wt'], colors['sert']], size=4)
ax2.set(ylabel='5-HT modulated neurons (%)', xlabel='', xticklabels=['Wild type\ncontrol', 'Sert-Cre'],
        ylim=[0, 80])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.pdf'))
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.png'))


