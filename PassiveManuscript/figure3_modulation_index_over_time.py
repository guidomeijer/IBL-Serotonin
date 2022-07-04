# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:50:06 2022

@author: Guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from serotonin_functions import paths, combine_regions, load_subjects, figure_style, high_level_regions

# Settings
MIN_NEURONS = 20

# Load in results
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure3')
mod_idx_df = pd.read_pickle(join(save_path, 'mod_over_time.pickle'))
#mod_idx_df['full_region'] = combine_regions(mod_idx_df['region'], abbreviate=True)
mod_idx_df['full_region'] = high_level_regions(mod_idx_df['region'])
mod_idx_df = mod_idx_df[mod_idx_df['full_region'] != 'root']
time_ax = mod_idx_df['time'].mean()

# Only include sert mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    mod_idx_df.loc[mod_idx_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
mod_idx_df = mod_idx_df[mod_idx_df['sert-cre'] == 1]

# Make into long form dataframe
mod_long_df = pd.DataFrame()
for i, region in enumerate(np.unique(mod_idx_df['full_region'])):
    if mod_idx_df[mod_idx_df['full_region'] == region].shape[0] < MIN_NEURONS:
        continue
    for ind in mod_idx_df[mod_idx_df['full_region'] == region].index.values:
        mod_long_df = pd.concat((mod_long_df, pd.DataFrame(data={
            'time': time_ax, 'abs_mod_idx': np.abs(mod_idx_df.loc[ind, 'mod_idx']),
            'region': region})), ignore_index=True)

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='abs_mod_idx', data=mod_long_df, hue='region', ci=68, ax=ax1,
             palette=colors)
ax1.set(xlim=[-1, 4], ylim=[0.05, 0.25], ylabel='Absolute modulation index', xlabel='Time (s)')
leg = ax1.legend(title='', bbox_to_anchor=(0.9, 0.45, 0.2, 0.4), prop={'size': 5})
leg.get_frame().set_linewidth(0.0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'modulation_index_over_time.pdf'))

