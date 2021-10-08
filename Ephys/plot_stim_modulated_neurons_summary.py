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
from serotonin_functions import paths, figure_style, get_full_region_name, load_subjects

# Settings
N_BINS = 30
ARTIFACT_CUTOFF = 0.6
AXIS_LIM = 0.8

# Paths
_, fig_path, save_path = paths()
fig_path = join(fig_path)
save_path = join(save_path)

# Load in results
all_neurons = pd.read_csv(join(save_path, 'stim_modulated_neurons.csv'))
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Exclude artifact neurons
light_neurons = light_neurons[light_neurons['roc_auc'] > ARTIFACT_CUTOFF]
all_neurons = pd.merge(all_neurons, light_neurons, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

# Add expression
subjects = pd.read_csv(join('..', 'subjects.csv'))
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

all_neurons['enhanced'] = all_neurons['stim_modulated'] & (all_neurons['stim_mod_roc'] > 0)
all_neurons['suppressed'] = all_neurons['stim_modulated'] & (all_neurons['stim_mod_roc'] < 0)

# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3.5), dpi=dpi)

ax1.hist(all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['stim_modulated'] == 0), 'stim_mod_roc'],
         6, density=False, histtype='bar', color=colors['wt'])
ax1.hist([all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['enhanced'] == 1), 'stim_mod_roc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 1) & (all_neurons['suppressed'] == 1), 'stim_mod_roc']],
         N_BINS, density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax1.set(xlim=[-.6, .6], xlabel='Modulation index', ylabel='Neuron count', title='SERT', xticks=np.arange(-0.6, 0.61, 0.3),
        ylim=[10**-1, 10**3], yscale='log', yticks=[10**-1, 10**0, 10**1, 10**2, 10**3], yticklabels=[0, 1, 10, 100, 1000])

ax2.hist([all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['enhanced'] == 1), 'stim_mod_roc'],
          all_neurons.loc[(all_neurons['sert-cre'] == 0) & (all_neurons['suppressed'] == 1), 'stim_mod_roc']],
         int(N_BINS/2), density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax2.set(xlim=[-0.6, 0.6], xlabel='Modulation index', ylabel='Neuron count', title='WT', ylim=[0, 5],
        xticks=np.arange(-0.6, 0.61, 0.3))

all_neurons['stim_modulated'] = all_neurons['stim_modulated'].astype(int)
summary_df = all_neurons.groupby('subject').sum()
summary_df['n_neurons'] = all_neurons.groupby('subject').size()
summary_df['perc_mod'] = (summary_df['stim_modulated'] / summary_df['n_neurons']) * 100
summary_df['sert-cre'] = (summary_df['sert-cre'] > 0).astype(int)

sns.swarmplot(x='sert-cre', y='perc_mod', data=summary_df, ax=ax3,
              palette=[colors['wt'], colors['sert']])
ax3.set(ylabel='Modulated neurons (%)', xlabel='', xticklabels=['Wild type\ncontrol', 'Sert-Cre'], ylim=[0, 20])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'opto_task_modulation_summary.pdf'))
plt.savefig(join(fig_path, 'Ephys', 'opto_task_modulation_summary.png'))