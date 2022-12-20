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
from serotonin_functions import paths, load_subjects, figure_style, combine_regions

# Settings
MIN_NEURONS = 10

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

# Make into long form dataframe
mod_long_df = pd.DataFrame()
for i, region in enumerate(np.unique(mod_idx_df['full_region'])):
    if mod_idx_df[mod_idx_df['full_region'] == region].shape[0] < MIN_NEURONS:
        continue
    for ind in mod_idx_df[mod_idx_df['full_region'] == region].index.values:
        mod_long_df = pd.concat((mod_long_df, pd.DataFrame(data={
            'time': time_ax, 'mod_idx': mod_idx_df.loc[ind, 'mod_idx'],
            'region': region})), ignore_index=True)

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -0.1), 1, 0.2, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='mod_idx', data=mod_long_df[(mod_long_df['region'] == 'mPFC')],
             errorbar='se', ax=ax1, color='k')
ax1.set(xlim=[-1, 3], ylim=[-0.04, 0.101], ylabel='Modulation index', xlabel='Time (s)')
#leg = ax1.legend(title='', bbox_to_anchor=(0.9, 0.45, 0.2, 0.4), prop={'size': 5})
#leg.get_frame().set_linewidth(0.0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(f_path, 'modulation_index_over_time.pdf'))

