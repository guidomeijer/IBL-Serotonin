# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:28:03 2022

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from serotonin_functions import (paths, load_subjects, figure_style, combine_regions,
                                 plot_scalar_on_slice)
from ibllib.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)

AP = [2, -1.5, -3.5]

# Load in results
fig_path, save_path = paths(dropbox=True)
f_path = join(fig_path, 'PaperPassive', 'figure4')
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))
mod_idx_df['full_region'] = combine_regions(mod_idx_df['region'], abbreviate=True)
#mod_idx_df['full_region'] = high_level_regions(mod_idx_df['region'])
mod_idx_df = mod_idx_df[mod_idx_df['full_region'] != 'root']
time_ax = mod_idx_df['time'].mean()

# Only include sert mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    mod_idx_df.loc[mod_idx_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
mod_idx_df = mod_idx_df[mod_idx_df['sert-cre'] == 1]

mod_idx_df.groupby(['region', 'time']).mean(numeric_only=True)

# %%

colors, dpi = figure_style()
CMAP = 'Spectral_r'
CLIM = 0.18

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4), dpi=dpi)

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax1,
                     slice='coronal', coord=AP[0]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-CLIM, CLIM])
ax1.axis('off')
ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax2,
                     slice='coronal', coord=AP[1]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-CLIM, CLIM])
ax2.axis('off')
ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(reg_neurons['region'].values, reg_neurons['mod_late'].values, ax=ax3,
                     slice='coronal', coord=AP[2]*1000, brain_atlas=ba, cmap=CMAP, clevels=[-CLIM, CLIM])
ax3.axis('off')
ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.2])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Mean modulation index', rotation=270, labelpad=10)
cbar.ax.set_yticks([-CLIM, 0, CLIM])

plt.savefig(join(fig_path, 'brain_map_modulation.pdf'))