# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:25:01 2022

@author: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import paths, load_subjects, figure_style

# Settings
asy_tb = 8
DIAGONALS = 4

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
Amyg_mPFC = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'mPFC-Amyg', 'r_opto'].to_numpy())
Amyg_ORB = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'ORB-Amyg', 'r_opto'].to_numpy())

# Get time axis
time_ax = jpecc_df['time'].mean()

# Calculate asymmetry
asy_Amyg_mPFC = np.empty((Amyg_mPFC.shape[2], Amyg_mPFC.shape[0] - (asy_tb*2-1)))
for j in range(Amyg_mPFC.shape[2]):
    for i, k in enumerate(range(asy_tb, Amyg_mPFC.shape[0] - (asy_tb - 1))):
        Amyg_mPFC_slice = Amyg_mPFC[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_Amyg_mPFC[j, i] = np.mean(Amyg_mPFC_slice[np.triu_indices(Amyg_mPFC_slice.shape[0], k=1)]
                                 - Amyg_mPFC_slice[np.tril_indices(Amyg_mPFC_slice.shape[0], k=-1)])

asy_Amyg_ORB = np.empty((Amyg_ORB.shape[2], Amyg_ORB.shape[0] - (asy_tb*2-1)))
for j in range(Amyg_ORB.shape[2]):
    for i, k in enumerate(range(asy_tb, Amyg_ORB.shape[0] - (asy_tb - 1))):
        Amyg_ORB_slice = Amyg_ORB[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_Amyg_ORB[j, i] = np.mean(Amyg_ORB_slice[np.triu_indices(Amyg_ORB_slice.shape[0], k=1)]
                                   - Amyg_ORB_slice[np.tril_indices(Amyg_ORB_slice.shape[0], k=-1)])
asy_time = jpecc_df['time'].mean()[range(asy_tb, Amyg_ORB.shape[0] - (asy_tb - 1))]

# Take diagonal lines (collapse over both time axes and take average)
# Trigger warning: this is extremely ugly code
diag_df = pd.DataFrame()
for j in range(Amyg_mPFC.shape[2]):
    for i in np.arange(-DIAGONALS, DIAGONALS+1):
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(Amyg_mPFC[:,:,j], offset=i),
            'diag_bl': (np.diagonal(Amyg_mPFC[:,:,j], offset=i)
                        - np.median(np.diagonal(Amyg_mPFC[:,:,j], offset=i)[time_ax[np.abs(i):] < 0])),
            'subject': j,
            'region_pair': 'Amyg-mPFC',
            'time': time_ax[np.abs(i):]})))
        if i == 0:
            this_time = time_ax
        else:
            this_time = time_ax[:-np.abs(i)]
        this_times = time_ax[:-np.abs(i)]
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(Amyg_mPFC[:,:,j], offset=i),
            'diag_bl': (np.diagonal(Amyg_mPFC[:,:,j], offset=i)
                        - np.median(np.diagonal(Amyg_mPFC[:,:,j], offset=i)[this_time < 0])),
            'subject': j,
            'region_pair': 'Amyg-mPFC',
            'time': this_time})))
for j in range(Amyg_ORB.shape[2]):
    for i in np.arange(-DIAGONALS, DIAGONALS+1):
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(Amyg_ORB[:,:,j], offset=i),
            'diag_bl': (np.diagonal(Amyg_ORB[:,:,j], offset=i)
                        - np.median(np.diagonal(Amyg_ORB[:,:,j], offset=i)[time_ax[np.abs(i):] < 0])),
            'subject': j,
            'region_pair': 'Amyg-ORB',
            'time': time_ax[np.abs(i):]})))
        if i == 0:
            this_time = time_ax
        else:
            this_time = time_ax[:-np.abs(i)]
        this_times = time_ax[:-np.abs(i)]
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(Amyg_ORB[:,:,j], offset=i),
            'diag_bl': (np.diagonal(Amyg_ORB[:,:,j], offset=i)
                        - np.median(np.diagonal(Amyg_ORB[:,:,j], offset=i)[this_time < 0])),
            'subject': j,
            'region_pair': 'Amyg-ORB',
            'time': this_time})))
diag_df = diag_df.groupby(['time', 'subject', 'region_pair']).mean()
diag_df = diag_df.melt(ignore_index=False).reset_index()


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(3.5, 1.75),
                                    gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)

ax1.imshow(np.flipud(np.mean(Amyg_mPFC, axis=2)), vmin=0.2, vmax=0.6, cmap='inferno',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax1.invert_xaxis()
ax1.plot([0, 0], [-1, 2], color='white')
ax1.plot([-1, 2], [0, 0], color='white')
ax1.plot([-1, 2], [-1, 2], color='white')
ax1.set(ylabel='Amyg', xlabel='mPFC, time from stim. (s)', title='jPECC: Amyg vs mPFC',
        xlim=[-1, 2], ylim=[-1, 2])

ax2.imshow(np.flipud(np.mean(Amyg_ORB, axis=2)), vmin=0.2, vmax=0.6, cmap='inferno',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax2.invert_xaxis()
ax2.plot([0, 0], [-1, 2], color='white')
ax2.plot([-1, 2], [0, 0], color='white')
ax2.plot([-1, 2], [-1, 2], color='white')
ax2.set(ylabel='Amyg', xlabel='mPFC, time from stim. (s)', title='Amyg vs ORB',
        xlim=[-1, 2], ylim=[-1, 2])

ax_cb.axis('off')
plt.tight_layout()

cb_ax = f.add_axes([0.8, 0.3, 0.01, 0.5])
cbar = f.colorbar(mappable=ax2.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=10)

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 3, color='royalblue', alpha=0.25, lw=0))
ax1.plot([-1, 2], [0, 0], ls='--', color='grey')
ax1.plot(asy_time, np.mean(asy_Amyg_mPFC, axis=0), color=colors['M2-mPFC'], label='Amyg-mPFC')
ax1.fill_between(asy_time,
                 np.mean(asy_Amyg_mPFC, axis=0)-(np.std(asy_Amyg_mPFC, axis=0)/np.sqrt(asy_Amyg_mPFC.shape[0])),
                 np.mean(asy_Amyg_mPFC, axis=0)+(np.std(asy_Amyg_mPFC, axis=0)/np.sqrt(asy_Amyg_mPFC.shape[0])),
                 alpha=0.2, color=colors['M2-mPFC'])
ax1.plot(asy_time, np.mean(asy_Amyg_ORB, axis=0), color=colors['M2-ORB'], label='Amyg-ORB')
ax1.fill_between(asy_time,
                 np.mean(asy_Amyg_ORB, axis=0)-(np.std(asy_Amyg_ORB, axis=0)/np.sqrt(asy_Amyg_ORB.shape[0])),
                 np.mean(asy_Amyg_ORB, axis=0)+(np.std(asy_Amyg_ORB, axis=0)/np.sqrt(asy_Amyg_ORB.shape[0])),
                 alpha=0.2, color=colors['M2-ORB'])
ax1.legend(prop={'size': 6}, frameon=True)
ax1.set(ylabel='jPECC asymmetry', xlabel='Time from stim. onset (s)',
        ylim=[-0.15, 0.15], xlim=[-1, 2], yticks=np.arange(-0.15, 0.16, 0.05))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'jPECC_asymmetry_Amyg.pdf'))

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.5, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag'], hue='region_pair', ax=ax1, ci=68,
             hue_order=['Amyg-ORB', 'Amyg-mPFC'], palette=[colors['M2-ORB'], colors['M2-mPFC']])
ax1.set(xlabel='Time (s)', ylabel='Canonical correlation (r)', xlim=[-1, 2], ylim=[0, 0.5])
ax1.legend(prop={'size': 6})

ax2.plot([-1, 2], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.2), 1, 0.5, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag_bl'], hue='region_pair', ax=ax2, ci=68,
             hue_order=['Amyg-ORB', 'Amyg-mPFC'], palette=[colors['M2-ORB'], colors['M2-mPFC']])
ax2.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 2], ylim=[-0.2, 0.3])
ax2.legend(frameon=True, prop={'size': 6}, loc='upper left')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA_Amyg.pdf'))

