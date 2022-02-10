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

# Paths
_, fig_path, save_path = paths()
map_path = join(fig_path, 'Ephys', 'BrainMaps')

# Load in results
all_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))

# Add genotype
subjects = load_subjects()
for i, nickname in enumerate(np.unique(all_neurons['subject'])):
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
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3.5), dpi=dpi)

ax1.hist(all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated_early'] == 0), 'mod_index_early'],
         10, density=False, histtype='bar', color=colors['wt'])
ax1.hist([all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['enhanced_early'] == 1), 'mod_index_early'],
          all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['suppressed_early'] == 1), 'mod_index_early']],
         N_BINS, density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax1.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count', title='Early stim', ylim=[10**-1, 10**3],
        xticks=np.arange(-1, 1.1, 0.5), yscale='log', yticks=[10**-1, 10**0, 10**1, 10**2, 10**3],
        yticklabels=[0, 1, 10, 100, 1000])


ax2.hist(all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['modulated_late'] == 0), 'mod_index_late'],
         10, density=False, histtype='bar', color=colors['wt'])
ax2.hist([all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['enhanced_late'] == 1), 'mod_index_late'],
          all_neurons.loc[(all_neurons['expression'] == 1) & (all_neurons['suppressed_late'] == 1), 'mod_index_late']],
         N_BINS, density=False, histtype='bar', stacked=True,
         color=[colors['enhanced'], colors['suppressed']])
ax2.set(xlim=[-1, 1], xlabel='Modulation index', ylabel='Neuron count', title='Late stim', yscale='log',
        ylim=[10**-1, 10**3], yticks=[10**-1, 10**0, 10**1, 10**2, 10**3], xticks=np.arange(-1, 1.1, 0.5),
        yticklabels=[0, 1, 10, 100, 1000])

summary_df = all_neurons.groupby('subject').sum()
summary_df['n_neurons'] = all_neurons.groupby('subject').size()
summary_df['perc_mod'] = (summary_df['modulated'] / summary_df['n_neurons']) * 100
summary_df['expression'] = (summary_df['expression'] > 0).astype(int)

sns.swarmplot(x='expression', y='perc_mod', data=summary_df, ax=ax3,
              palette=[colors['wt'], colors['sert']])
ax3.set(ylabel='Modulated neurons (%)', xlabel='', xticklabels=['Wild type\ncontrol', 'Sert-Cre'], ylim=[0, 80])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.pdf'))
plt.savefig(join(fig_path, 'Ephys','opto_modulation_summary.png'))


