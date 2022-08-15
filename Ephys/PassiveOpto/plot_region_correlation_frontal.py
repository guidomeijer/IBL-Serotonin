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

# Paths
fig_path, save_path = paths()

# Load in data
corr_df = pd.read_csv(join(save_path, 'region_corr_frontal.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    corr_df.loc[corr_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

corr_df = corr_df.groupby(['subject', 'time', 'region_pair']).mean().reset_index()

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.add_patch(Rectangle((0, 0), 1, 4, color='royalblue', alpha=0.25, lw=0))
#sns.lineplot(x='time', y='region_corr', ax=ax1, legend='brief', hue='region_pair', estimator=None,
#             units='subject', data=corr_df[corr_df['sert-cre'] == 1])
sns.lineplot(x='time', y='region_corr', ax=ax1, legend='brief', hue='region_pair', ci=68,
             data=corr_df[corr_df['sert-cre'] == 1])
ax1.set(ylabel='Correlation (r)', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3])
leg = ax1.legend(title='', prop={'size': 4}, loc='upper right', frameon=False)
plt.tight_layout()
sns.despine(trim=True)

"""
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
"""
