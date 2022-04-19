#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS = 20

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive')

# Load in results
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'], split_thalamus=False)
#light_neurons['full_region'] = light_neurons['region']

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Add enhanced and suppressed
light_neurons['enhanced_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] > 0)
light_neurons['suppressed_late'] = light_neurons['modulated'] & (light_neurons['mod_index_late'] < 0)
light_neurons['enhanced_early'] = light_neurons['modulated'] & (light_neurons['mod_index_early'] > 0)
light_neurons['suppressed_early'] = light_neurons['modulated'] & (light_neurons['mod_index_early'] < 0)

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Calculate summary statistics
summary_df = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).sum()
summary_df['n_neurons'] = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).size()
summary_df['modulation_index'] = light_neurons[light_neurons['expression'] == 1].groupby(['full_region']).mean()['mod_index_late']
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS]
# Get ordered regions
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()



# %% Plot

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.75, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=summary_df.sort_values('perc_mod', ascending=False),
            color=colors['general'], ax=ax1)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 50], xticks=np.arange(0, 51, 10))
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'figure2_perc_light_modulated_neurons_per_region.pdf'))
