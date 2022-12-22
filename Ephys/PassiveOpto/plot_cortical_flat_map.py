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
from ibllib.atlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice
from ibllib.atlas import FlatMap
from ibllib.atlas import AllenAtlas

# Settings
RESOLUTION = 25
DEPTH = 600
MIN_NEURONS = 40

# Initialize
flmap = FlatMap(flatmap='dorsal_cortex', res_um=RESOLUTION)
ba = AllenAtlas(res_um=RESOLUTION)

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
sert_neurons = all_neurons[all_neurons['expression'] == 1]

# Get percentage modulated per region
reg_neurons = ((sert_neurons.groupby('region').sum()['modulated'] / sert_neurons.groupby('region').size()) * 100).to_frame()
reg_neurons = reg_neurons.rename({0: 'percentage'}, axis=1)
reg_neurons['mod_early'] = sert_neurons.groupby('region').mean()['mod_index_early']
reg_neurons['mod_late'] = sert_neurons.groupby('region').mean()['mod_index_late']
reg_neurons['latency'] = sert_neurons[sert_neurons['modulated'] == 1].groupby('region').median()['latency_peak_onset'] * 1000
reg_neurons['n_neurons'] = sert_neurons.groupby(['region']).size()
reg_neurons['n_mod_neurons'] = sert_neurons[sert_neurons['modulated'] == 1].groupby(['region']).size()
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['region'] != 'root']


# %%

colors, dpi = figure_style()

# Plot brain map slices
f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
f, ax, cbar = plot_scalar_on_flatmap(reg_neurons['region'], reg_neurons['mod_late'], depth=DEPTH, mapping='Beryl',
                                     hemisphere='left', clevels=[-0.05, 0.001], background='boundary', show_cbar=True,
                                     cmap='plasma', flmap_atlas=flmap, ax=ax)
ax.set_axis_off()

"""
# %%
f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
f, ax, cbar = plot_scalar_on_flatmap(['RSPv','RSPd'], np.array([1, 2]), depth=DEPTH, mapping='Beryl',
                                     hemisphere='left', background='boundary', show_cbar=True,
                                     cmap='plasma', flmap_atlas=flmap, ax=ax)
ax.set_axis_off()
"""

# %%
fig, ax, cbar = plot_scalar_on_slice(reg_neurons['region'], reg_neurons['mod_late'], slice='top',
                                     mapping='Beryl', hemisphere='left', clevels=[-0.05, 0.01],
                                     background='boundary', cmap='plasma', brain_atlas=ba,
                                     show_cbar=True)
ax.set_axis_off()