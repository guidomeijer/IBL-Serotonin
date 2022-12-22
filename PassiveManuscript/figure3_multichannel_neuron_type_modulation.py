#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:49:47 2022
By: Guido Meijer
"""

from os.path import join
from serotonin_functions import paths, figure_style, load_subjects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from statsmodels.stats.oneway import anova_oneway
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

# Settings
var = 'mod_index_late'
MIN_NEURONS = 3

# Get paths
fig_dir, data_dir = paths(dropbox=True)
fig_dir = join(fig_dir, 'PaperPassive', 'figure3')

# Load in data
light_neurons = pd.read_csv(join(data_dir, 'light_modulated_neurons.csv'))
neuron_type = pd.read_csv(join(data_dir, 'neuron_type_multichannel.csv'))
all_neurons = pd.merge(light_neurons, neuron_type, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])

# Select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]
all_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Drop neurons that could not be classified into a group
all_neurons = all_neurons[(all_neurons['type'] != 'Und.') & (~all_neurons['type'].isnull())]

# Only MOs for now
#all_neurons = all_neurons[all_neurons['region'] != 'MOs']

# Get percentage of modulated neurons per animal per neuron type
perc_mod = ((all_neurons.groupby(['subject', 'type']).sum(numeric_only=True)['modulated']
            / all_neurons.groupby(['subject', 'type']).size()) * 100).to_frame()
perc_mod['n_neurons'] = all_neurons.groupby(['subject', 'type']).size()
perc_mod = perc_mod.rename(columns={0: 'percentage'}).reset_index()
perc_mod = perc_mod[perc_mod['n_neurons'] >= MIN_NEURONS]

# Select only modulated neurons
all_neurons = all_neurons[all_neurons['modulated']]

# Run ANOVA
mod = ols(f'{var} ~ type', data=all_neurons).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(all_neurons[var], all_neurons['type'])
tukey_mod = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA modulation p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_mod)

mod = ols('percentage ~ type', data=perc_mod).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(perc_mod['percentage'], perc_mod['type'])
tukey_perc = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA percentage p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_perc)

# %% Plot modulation
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

sns.barplot(data=perc_mod, x='type', y='percentage', errorbar='se', order=['NS', 'RS1', 'RS2'],
            palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax1)
ax1.set(ylim=[0, 61], ylabel='Modulated neurons (%)', xlabel='')

#sns.violinplot(data=all_neurons, x='type', y=var, order=['NS', 'RS1', 'RS2'],
#               palette=[colors['NS'], colors['RS1'], colors['RS2']], linewidths=0, ax=ax1)
sns.swarmplot(data=all_neurons, x='type', y=var, order=['NS', 'RS1', 'RS2'], legend=None,
              size=3, hue='type', palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax2)
ax2.set(ylabel='Modulation index', ylim=[-0.5, 0.5], yticks=[-0.5, 0, 0.5], xlabel='')


plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_dir, 'perc_type_mod.pdf'))
