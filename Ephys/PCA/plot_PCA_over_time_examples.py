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
from serotonin_functions import figure_style
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from serotonin_functions import paths, remap

# Settings
SUBJECTS = ['ZFM-03330', 'ZFM-02600', 'ZFM-02601', 'ZFM-01802']
DATES = ['2022-02-16', '2021-08-26', '2021-11-18', '2021-03-11']
REGIONS = ['mPFC', 'ORB', 'Thal', 'Amyg']

# Paths 
fig_path, save_path = paths()
fig_path = join(fig_path, 'PCA')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))

colors, dpi = figure_style()
f, axs = plt.subplots(1, 5, figsize=(8, 1.75), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.02]}, dpi=dpi)
for i in range(len(SUBJECTS)):
    df_slice = pca_df[(pca_df['subject'] == SUBJECTS[i]) & (pca_df['date'] == DATES[i])
                      & (pca_df['region'] == REGIONS[i])]
    sp = axs[i].scatter(df_slice['pca1'], df_slice['pca2'], c=df_slice['time'], cmap='twilight_r')
    axs[i].axis('off')
    axs[i].set(xlabel='PC 1', ylabel='PC 2', title=REGIONS[i])

axs[4].axis('off')
cb_ax = f.add_axes([0.9, 0.2, 0.01, 0.6])
cbar = plt.colorbar(sp, cax=cb_ax)
cbar.ax.set_ylabel('Time (s)', rotation=270, labelpad=10)
cbar.ax.set_yticks([-1, 0, 1, 2, 3])

#plt.colorbar(sp)

sns.despine(trim=True)
#plt.tight_layout()

#plt.savefig(join(fig_path, 'example_trajectories.jpg'), dpi=300)

 
