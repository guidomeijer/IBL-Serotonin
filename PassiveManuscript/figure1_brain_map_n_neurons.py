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
from serotonin_functions import paths, figure_style, load_subjects, plot_scalar_on_slice
from ibllib.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)

# Settings
HISTOLOGY = True
N_BINS = 30
MIN_NEURONS = 0
AP = [2, -1.5, -3.5]

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure1')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Get percentage modulated per region
reg_neurons = ((all_neurons.groupby('region').sum()['modulated'] / all_neurons.groupby('region').size()) * 100).to_frame()
reg_neurons = reg_neurons.rename({0: 'percentage'}, axis=1)
reg_neurons['mod_early'] = all_neurons.groupby('region').median()['mod_index_early']
reg_neurons['mod_late'] = all_neurons.groupby('region').median()['mod_index_late']
reg_neurons['latency'] = all_neurons[all_neurons['modulated'] == 1].groupby('region').median()['latency_peak_hw'] * 1000
reg_neurons['n_neurons'] = all_neurons.groupby(['region']).size()
reg_neurons['n_mod_neurons'] = all_neurons[all_neurons['modulated'] == 1].groupby(['region']).size()
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['region'] != 'root']


# %%

colors, dpi = figure_style()

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['n_neurons'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 200])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['n_neurons'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 200])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['n_neurons'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 200])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.2])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('# recorded neurons', rotation=270, labelpad=10)
#cbar.ax.set_yticks([0, 25, 50])

plt.savefig(join(fig_path, 'brain_map_n_neurons.pdf'))

