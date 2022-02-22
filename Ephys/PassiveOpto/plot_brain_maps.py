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
MIN_NEURONS = 10
AP = [2, -1.5, -3.5]

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
reg_neurons = (sert_neurons.groupby('region').sum()['modulated'] / sert_neurons.groupby('region').size() * 100).to_frame()
reg_neurons = reg_neurons.rename({0: 'percentage'}, axis=1)
reg_neurons['mod_early'] = sert_neurons.groupby('region').median()['mod_index_early']
reg_neurons['mod_late'] = sert_neurons.groupby('region').median()['mod_index_late']
reg_neurons['latency'] = sert_neurons.groupby('region').median()['latency'] * 1000
reg_neurons['n_neurons'] = sert_neurons.groupby(['region']).size()
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['region'] != 'root']


# %%

colors, dpi = figure_style()

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax2.axis('off')
ax1.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax3.axis('off')
ax1.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('% modulated neurons', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'perc_mod_neurons.jpg'), dpi=300)
plt.savefig(join(map_path, 'perc_mod_neurons.pdf'))

# %%
# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_early'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])

ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_early'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax2.axis('off')
ax1.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_early'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax3.axis('off')
ax1.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Modulation index', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'modulation_index_early.jpg'), dpi=300)
plt.savefig(join(map_path, 'modulation_index_early.pdf'))

# %%

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax2.axis('off')
ax1.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax3.axis('off')
ax1.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Modulation index', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'modulation_index_late.jpg'), dpi=300)
plt.savefig(join(map_path, 'modulation_index_late.pdf'))

# %%

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].abs().values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 0.4])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].abs().values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 0.4])
ax2.axis('off')
ax1.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].abs().values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 0.4])
ax3.axis('off')
ax1.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Abs. modulation index', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'modulation_index_late_abs.jpg'), dpi=300)
plt.savefig(join(map_path, 'modulation_index_late_abs.pdf'))

# %%

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['latency'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap='plasma', clevels=[0, 600])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['latency'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap='plasma', clevels=[0, 600])
ax2.axis('off')
ax1.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['latency'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap='plasma', clevels=[0, 600])
ax3.axis('off')
ax1.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Latency (ms)', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'latency_map.jpg'), dpi=300)
plt.savefig(join(map_path, 'latency_map.pdf'))




