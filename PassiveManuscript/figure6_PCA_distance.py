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
from scipy.stats import kruskal
from matplotlib.patches import Rectangle
from serotonin_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Settings
REGIONS = ['M2', 'OFC', 'mPFC']

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure6')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))
pca_dist_df = pd.read_csv(join(save_path, 'pca_dist_regions.csv'))
pca_dist_df.loc[pca_dist_df['region'] == 'ORB', 'region'] = 'OFC'

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    pca_df.loc[pca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    pca_dist_df.loc[pca_dist_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
pca_dist_df = pca_dist_df[(pca_dist_df['sert-cre'] == 1) & (pca_dist_df['region'].isin(REGIONS))]

# Take average over multiple sessions per animal
pca_dist_df = pca_dist_df.groupby(['subject', 'time', 'region']).median().reset_index()

# Smooth traces (this is ugly I know)
for i, subject in enumerate(pca_dist_df['subject'].unique()):
    for r, region in enumerate(REGIONS):
        if pca_dist_df.loc[(pca_dist_df['subject'] == subject) & (pca_dist_df['region'] == region)].shape[0] > 0:
            pca_dist_df.loc[(pca_dist_df['subject'] == subject) & (pca_dist_df['region'] == region), 'pca_dist_smooth'] = (
                smooth_interpolate_signal_sg(pca_dist_df.loc[(pca_dist_df['subject'] == subject)
                                                             & (pca_dist_df['region'] == region), 'pca_dist'], window=15))

# Do statistics
pca_table_df = pca_dist_df.pivot(index='time', columns=['region', 'subject'], values='pca_dist')
pca_table_df = pca_table_df.reset_index()
for i in pca_table_df.index.values:
    pca_table_df.loc[i, 'p_value'] = kruskal(pca_table_df.loc[i, 'M2'], pca_table_df.loc[i, 'mPFC'],
                                             pca_table_df.loc[i, 'OFC'])[1]

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist_smooth', ax=ax1, legend='brief', hue='region', ci=68,
             data=pca_dist_df, units='subject', estimator=None,
             hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
#ax1.plot(pca_table_df.loc[(pca_table_df['p_value'] < 0.05) & (pca_table_df['time'] < 1), 'time'],
#         np.ones(pca_table_df[(pca_table_df['p_value'] < 0.05) & (pca_table_df['time'] < 1)].shape[0])*4, color='k')
ax1.set(xlim=[-1, 2], xlabel='Time (s)', ylabel='PCA traj. displacement (a.u.)',
        xticks=[-1, 0, 1, 2], ylim=[0, 4.05])
leg = ax1.legend(title='', frameon=True, bbox_to_anchor=(0.45, 0.6), prop={'size': 5.5})
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'PCA_traj_distance_per_subject.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='pca_dist_smooth', ax=ax1, legend='brief', hue='region', ci=68,
             data=pca_dist_df, hue_order=REGIONS, palette=[colors[i] for i in REGIONS])
#ax1.plot(pca_table_df.loc[(pca_table_df['p_value'] < 0.05) & (pca_table_df['time'] < 1), 'time'],
#         np.ones(pca_table_df[(pca_table_df['p_value'] < 0.05) & (pca_table_df['time'] < 1)].shape[0])*4, color='k')
ax1.set(xlim=[-1, 2], xlabel='Time (s)', ylabel='PCA traj. displacement (a.u.)',
        xticks=[-1, 0, 1, 2], ylim=[0.5, 2.5])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = [f'M2 (n={pca_dist_df[pca_dist_df["region"] == "M2"]["subject"].unique().size})',
              f'OFC (n={pca_dist_df[pca_dist_df["region"] == "OFC"]["subject"].unique().size})',
              f'mPFC (n={pca_dist_df[pca_dist_df["region"] == "mPFC"]["subject"].unique().size})']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 4}, loc='upper right')
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'PCA_traj_distance.pdf'))

