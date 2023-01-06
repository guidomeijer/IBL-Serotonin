#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:45:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from os.path import join
from serotonin_functions import paths, figure_style, combine_regions, load_subjects

# Settings
MIN_NEURONS_POOLED = 1
MIN_NEURONS_PER_MOUSE = 1
MIN_MOD_NEURONS = 1
MIN_REC = 1

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in results
awake_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
anes_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons_anesthesia.csv'))
light_neurons = pd.merge(awake_neurons, anes_neurons, on=['pid', 'eid', 'subject', 'neuron_id',
                                                        'date', 'probe', 'region'])

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    light_neurons.loc[light_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get full region names
light_neurons['full_region'] = combine_regions(light_neurons['region'])

# Drop root and void
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'root' in j])
light_neurons = light_neurons.reset_index(drop=True)
light_neurons = light_neurons.drop(index=[i for i, j in enumerate(light_neurons['full_region']) if 'void' in j])

# Get modulated neurons
mod_neurons = light_neurons[(light_neurons['sert-cre'] == 1)
                            & ((light_neurons['modulated_x'] == 1) & (light_neurons['modulated_y'] == 1))]
#mod_neurons = mod_neurons.groupby('full_region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)


# %% Plot percentage modulated neurons per region

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
(
 so.Plot(mod_neurons, x='mod_index_late_x', y='mod_index_late_y')
     .add(so.Dot(pointsize=2))
     #.add(so.Line(color='k', linewidth=1), so.PolyFit(order=1))
     .label(x='Modulation index awake', y='Modulation index anesthesia')
     .limit(x=[-1, 1], y=[-1, 1])
     .on(ax1)
     .plot()
 )
ax1.plot([-1, 1], [0, 0], color='k', ls='--')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'mod_index_awake_vs_anesthesia.pdf'))

