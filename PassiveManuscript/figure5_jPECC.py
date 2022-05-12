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
from serotonin_functions import paths, load_subjects, figure_style

# Settings
asy_tb = 5

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, 'jPECC_frontal.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    jpecc_df.loc[jpecc_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
jpecc_df = jpecc_df[jpecc_df['sert-cre'] == 1]
   
# Get 3D array of all jPECC
M2_mPFC = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'M2-mPFC', 'r_opto'].to_numpy())
M2_ORB = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'M2-ORB', 'r_opto'].to_numpy())

# Calculate asymmetry 
asy_M2_mPFC = np.empty((M2_mPFC.shape[2], M2_mPFC.shape[0] - (asy_tb*2-1)))
for j in range(M2_mPFC.shape[2]):
    for i, k in enumerate(range(asy_tb, M2_mPFC.shape[0] - (asy_tb - 1))):
        M2_mPFC_slice = M2_mPFC[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_M2_mPFC[j, i] = np.mean(M2_mPFC_slice[np.triu_indices(M2_mPFC_slice.shape[0], k=1)]
                                 - M2_mPFC_slice[np.tril_indices(M2_mPFC_slice.shape[0], k=-1)])
        
asy_M2_ORB = np.empty((M2_ORB.shape[2], M2_ORB.shape[0] - (asy_tb*2-1)))
for j in range(M2_ORB.shape[2]):
    for i, k in enumerate(range(asy_tb, M2_ORB.shape[0] - (asy_tb - 1))):
        M2_ORB_slice = M2_ORB[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_M2_ORB[j, i] = np.mean(M2_ORB_slice[np.triu_indices(M2_ORB_slice.shape[0], k=1)]
                                   - M2_ORB_slice[np.tril_indices(M2_ORB_slice.shape[0], k=-1)])
 

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(3.5, 1.75),
                                    gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)

ax1.imshow(np.flipud(np.mean(M2_mPFC, axis=2)), vmin=0.2, vmax=0.6, cmap='inferno')
ax1.set(title='M2-mPFC')

ax2.imshow(np.flipud(np.mean(M2_ORB, axis=2)), vmin=0.2, vmax=0.6, cmap='inferno')
ax2.set(title='M2-ORB')

ax_cb.axis('off')

cb_ax = f.add_axes([0.8, 0.2, 0.01, 0.6])
cbar = f.colorbar(mappable=ax2.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=10)

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
x_time = jpecc_df['time'].mean()[range(asy_tb, M2_ORB.shape[0] - (asy_tb - 1))]
ax1.plot(x_time, np.mean(asy_M2_mPFC, axis=0), color=colors['M2-mPFC'], label='M2-mPFC')
ax1.fill_between(x_time,
                 np.mean(asy_M2_mPFC, axis=0)-(np.std(asy_M2_mPFC, axis=0)/np.sqrt(asy_M2_mPFC.shape[0]))/2,
                 np.mean(asy_M2_mPFC, axis=0)+(np.std(asy_M2_mPFC, axis=0)/np.sqrt(asy_M2_mPFC.shape[0]))/2,
                 alpha=0.5, lw=0, color=colors['M2-mPFC'])
ax1.plot(x_time, np.mean(asy_M2_ORB, axis=0), color=colors['M2-ORB'], label='M2-ORB')
ax1.fill_between(x_time,
                 np.mean(asy_M2_ORB, axis=0)-(np.std(asy_M2_ORB, axis=0)/np.sqrt(asy_M2_ORB.shape[0]))/2,
                 np.mean(asy_M2_ORB, axis=0)+(np.std(asy_M2_ORB, axis=0)/np.sqrt(asy_M2_ORB.shape[0]))/2,
                 alpha=0.5, lw=0, color=colors['M2-ORB'])
ax1.legend()
ax1.set(ylabel='jPECC asymmetry', xlabel='Time from stim. onset (s)', xticks=[0, .5, 1, 1.5, 2],
        ylim=[-0.1, 0.1])

plt.tight_layout()
sns.despine(trim=True)

