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
from serotonin_functions import (paths, figure_style, load_subjects, plot_scalar_on_slice,
                                 combine_regions)
from ibllib.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)

# Settings
MIN_NEURONS = 10

# Paths
_, fig_path, save_path = paths()
map_path = join(fig_path, 'Ephys', 'BrainMaps')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_neurons['full_region'] = combine_regions(all_neurons['region'], split_thalamus=False)

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_neurons.loc[all_neurons['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
    all_neurons.loc[all_neurons['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Only sert-cre mice
sert_neurons = all_neurons[all_neurons['sert-cre'] == 1]

# Get percentage modulated per region
reg_neurons = sert_neurons.groupby('full_region').median()['latency'].to_frame()
reg_neurons['n_neurons'] = sert_neurons.groupby(['full_region']).size()
reg_neurons = reg_neurons.loc[reg_neurons['n_neurons'] >= MIN_NEURONS]
reg_neurons = reg_neurons.reset_index()
reg_neurons = reg_neurons[reg_neurons['full_region'] != 'root']


# %%

colors, dpi = figure_style()

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax1,
                     slice='coronal', coord=2000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax1.axis('off')
ax1.set(title='+2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax2,
                     slice='coronal', coord=-2000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax2.axis('off')
ax2.set(title='-2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['percentage'].values, ax=ax3,
                     slice='coronal', coord=-3000, brain_atlas=ba, cmap='YlOrRd', clevels=[0, 50])
ax3.axis('off')
ax3.set(title='-3 mm AP')

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
                     slice='coronal', coord=2000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])

ax1.axis('off')
ax1.set(title='+2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_early'].values, ax=ax2,
                     slice='coronal', coord=-2000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax2.axis('off')
ax2.set(title='-2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_early'].values, ax=ax3,
                     slice='coronal', coord=-3000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax3.axis('off')
ax3.set(title='-3 mm AP')

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
                     slice='coronal', coord=2000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax1.axis('off')
ax1.set(title='+2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax2,
                     slice='coronal', coord=-1400, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax2.axis('off')
ax2.set(title='-2 mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax3,
                     slice='coronal', coord=-3000, brain_atlas=ba, cmap='coolwarm', clevels=[-0.2, 0.2])
ax3.axis('off')
ax3.set(title='-3 mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.35, 0.01, 0.3])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Modulation index', rotation=270, labelpad=10)
plt.savefig(join(map_path, 'modulation_index_late.jpg'), dpi=300)
plt.savefig(join(map_path, 'modulation_index_late.pdf'))


# %% Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)

#ax2.hist(all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 0), 'mod_index_late'],
#         10, density=False, histtype='bar', color=colors['wt'])
ax1.hist([all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 1), 'mod_index_early'],
          all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated'] == 1), 'mod_index_late']],
         N_BINS, density=False, histtype='step', stacked=False,
         color=[colors['early'], colors['late']])
ax1.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count',
        xticks=np.arange(-1, 1.1, 0.5), ylim=[0, 100])
ax1.legend(['Early', 'Late'], frameon=False)

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


