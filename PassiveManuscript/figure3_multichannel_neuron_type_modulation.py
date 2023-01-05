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
MIN_NEURONS = 5

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

# Select only modulated neurons
mod_neurons = all_neurons[all_neurons['modulated']]

# %% Visual cortex
# Get percentage of modulated neurons per animal per neuron type
#vis_neurons = all_neurons[np.in1d(all_neurons['region'], ['VISa', 'VISam', 'VISp'])]
vis_neurons = all_neurons[np.in1d(all_neurons['region'], ['VISa', 'VISam', 'VISp', 'VISpm'])]
perc_mod = ((vis_neurons.groupby(['subject', 'type']).sum(numeric_only=True)['modulated']
            / vis_neurons.groupby(['subject', 'type']).size()) * 100).to_frame()
perc_mod['n_neurons'] = vis_neurons.groupby(['subject', 'type']).size()
perc_mod = perc_mod.rename(columns={0: 'percentage'}).reset_index()
perc_mod = perc_mod[perc_mod['n_neurons'] >= MIN_NEURONS]

# Select only modulated neurons
vis_neurons = vis_neurons[vis_neurons['modulated']]

# Run ANOVA
mod = ols(f'{var} ~ type', data=vis_neurons).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(vis_neurons[var], vis_neurons['type'])
tukey_mod = mc.tukeyhsd(alpha=0.05)
print(f'\nVisual cortex\nANOVA modulation p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_mod)

mod = ols('percentage ~ type', data=perc_mod).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(perc_mod['percentage'], perc_mod['type'])
tukey_perc = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA percentage p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_perc)

# %% M2 
# Get percentage of modulated neurons per animal per neuron type
mos_neurons = all_neurons[np.in1d(all_neurons['region'], ['MOs'])]
perc_mod_mos = ((mos_neurons.groupby(['subject', 'type']).sum(numeric_only=True)['modulated']
                 / mos_neurons.groupby(['subject', 'type']).size()) * 100).to_frame()
perc_mod_mos['n_neurons'] = mos_neurons.groupby(['subject', 'type']).size()
perc_mod_mos = perc_mod_mos.rename(columns={0: 'percentage'}).reset_index()
perc_mod_mos = perc_mod_mos[perc_mod_mos['n_neurons'] >= MIN_NEURONS]

# Select only modulated neurons
mos_neurons = mos_neurons[mos_neurons['modulated']]

# Run ANOVA
mod = ols(f'{var} ~ type', data=mos_neurons).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(mos_neurons[var], mos_neurons['type'])
tukey_mod = mc.tukeyhsd(alpha=0.05)
print(f'\nM2\nANOVA modulation p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_mod)

mod = ols('percentage ~ type', data=perc_mod_mos).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(perc_mod_mos['percentage'], perc_mod_mos['type'])
tukey_perc = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA percentage p = {aov_table.loc["type", "PR(>F)"]}\n')
print(tukey_perc)

# %%
# Merge the two dataframes
perc_mod['region'] = 'Visual cortex'
perc_mod_mos['region'] = 'M2'
perc_mod_merged = pd.concat((perc_mod, perc_mod_mos))

# %% Plot modulation
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
plt.subplots_adjust(wspace=2)

sns.barplot(data=perc_mod_merged, x='region', y='percentage', errorbar='se', hue='type',
            hue_order=['NS', 'RS1', 'RS2'],
            palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax1)
"""
sns.swarmplot(data=perc_mod_merged, x='region', y='percentage', hue='type',
              hue_order=['NS', 'RS1', 'RS2'], dodge=True, legend=False, size=3,
              palette=['gray', 'gray', 'gray'], ax=ax1)
"""
ax1.set(ylim=[0, 80], ylabel='Modulated neurons (%)', xlabel='')
ax1.legend(frameon=False, prop={'size': 5.5}, bbox_to_anchor=(0.6, 0.7))

sns.swarmplot(data=mod_neurons, x='type', y=var, order=['NS', 'RS1', 'RS2'], legend=None,
              size=3, hue='type', palette=[colors['NS'], colors['RS1'], colors['RS2']], ax=ax2)
ax2.set(ylabel='Modulation index', ylim=[-0.5, 0.5], yticks=[-0.5, 0, 0.5], xlabel='')


plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_dir, 'perc_type_mod.pdf'))
