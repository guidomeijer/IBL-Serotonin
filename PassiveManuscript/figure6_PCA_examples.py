#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from serotonin_functions import figure_style
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from serotonin_functions import paths, remap

# Settings
SUBJECTS = ['ZFM-02600', 'ZFM-02600', 'ZFM-03330']
DATES = ['2021-08-28', '2021-08-26', '2022-02-16']
REGIONS = ['M2', 'OFC', 'mPFC']
TITLES = ['Secondary motor', 'Orbitofrontal', 'Medial prefrontal']

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure6')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))

colors, dpi = figure_style()
f, axs = plt.subplots(1, 4, figsize=(5.25, 1.75), gridspec_kw={'width_ratios': [1, 1, 1, 0.02]}, dpi=dpi)
for i in range(len(SUBJECTS)):
    df_slice = pca_df[(pca_df['subject'] == SUBJECTS[i]) & (pca_df['date'] == DATES[i])
                      & (pca_df['region'] == REGIONS[i])]

    """
    points = np.array([df_slice['pca1'], df_slice['pca2']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(df_slice['time'].min(), df_slice['time'].max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(df_slice['time'])
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    """

    sp = axs[i].scatter(df_slice['pca1'], df_slice['pca2'], c=df_slice['time'], cmap='')
    axs[i].axis('off')
    axs[i].set(xlabel='PC 1', ylabel='PC 2', title=TITLES[i])

axs[3].axis('off')
cb_ax = f.add_axes([0.9, 0.2, 0.01, 0.6])
cbar = plt.colorbar(sp, cax=cb_ax)
cbar.ax.set_ylabel('Time (s)', rotation=270, labelpad=10)
cbar.ax.set_yticks([-1, 0, 1, 2, 3])

sns.despine(trim=True)
plt.savefig(join(fig_path, 'example_trajectories.pdf'))


