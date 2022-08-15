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
SERT_SUB = 'ZFM-03329'
SERT_DATE = '2022-03-03'

WT_SUB = 'ZFM-03324'
WT_DATE = '2022-04-13'

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Load in data
pca_df = pd.read_csv(join(save_path, 'pca_all_neurons.csv'))

colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(4, 1.75), gridspec_kw={'width_ratios': [1, 1, 0.15]}, dpi=dpi)

df_slice = pca_df[(pca_df['subject'] == SERT_SUB) & (pca_df['date'] == SERT_DATE)]
sp = ax1.scatter(df_slice['pca1'], df_slice['pca2'], c=df_slice['time'], cmap='twilight_r')
ax1.axis('off')
ax1.set(xlabel='PC 1', ylabel='PC 2', title='SERT')

df_slice = pca_df[(pca_df['subject'] == WT_SUB) & (pca_df['date'] == WT_DATE)]
sp = ax2.scatter(df_slice['pca1'], df_slice['pca2'], c=df_slice['time'], cmap='twilight_r')
ax2.axis('off')
ax2.set(xlabel='PC 1', ylabel='PC 2', title='WT')

ax_cb.axis('off')
cb_ax = f.add_axes([0.9, 0.2, 0.01, 0.6])
cbar = plt.colorbar(sp, cax=cb_ax)
cbar.ax.set_ylabel('Time (s)', rotation=270, labelpad=10)
cbar.ax.set_yticks([-1, 0, 1, 2, 3, 4])

#plt.colorbar(sp)

#sns.despine(trim=True)
#plt.tight_layout()

plt.savefig(join(fig_path, 'example_PCA_trajectories_all_neurons.pdf'))


