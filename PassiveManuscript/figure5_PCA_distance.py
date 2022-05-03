# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:25:01 2022

@author: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from 
from matplotlib.patches import Rectangle
from serotonin_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Settings
REGIONS = ['M2', 'ORB', 'mPFC']

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))
pca_dist_df = pd.read_csv(join(save_path, 'pca_dist_regions.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    pca_df.loc[pca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    pca_dist_df.loc[pca_dist_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    
# Smooth traces (this is ugly I know)
for i, pid in enumerate(pca_dist_df['pid'].unique()):
    for r, region in enumerate(REGIONS):
        if pca_dist_df.loc[(pca_dist_df['pid'] == pid) & (pca_dist_df['region'] == region)].shape[0] > 0:
            pca_dist_df.loc[(pca_dist_df['pid'] == pid) & (pca_dist_df['region'] == region), 'pca_dist_smooth'] = (
                smooth_interpolate_signal_sg(pca_dist_df.loc[(pca_dist_df['pid'] == pid)
                                                             & (pca_dist_df['region'] == region), 'pca_dist'], window=11))
    
    
# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
#ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist_smooth', ax=ax1, legend='brief', hue='region', ci=68,
             data=pca_dist_df[(pca_dist_df['sert-cre'] == 1) & (pca_dist_df['region'].isin(REGIONS))],
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
ax1.set(xlim=[-0.5, 1.5], xlabel='Time (s)', ylabel='PCA traj. displacement (a.u.)',
        xticks=[-0.5, 0, 0.5, 1, 1.5], ylim=[0, 4])
ax1.legend(title='', frameon=False, bbox_to_anchor=(0.55, 0.6))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'PCA_traj_distance.pdf'))

