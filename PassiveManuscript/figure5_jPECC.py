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
asy_tb = 6
DIAGONALS = 3

# Paths
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure5')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, 'jPECC.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    jpecc_df.loc[jpecc_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
jpecc_df = jpecc_df[jpecc_df['sert-cre'] == 1]

# Get 3D array of all jPECC
M2_mPFC = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'M2-mPFC', 'r_opto'].to_numpy())
M2_ORB = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == 'M2-ORB', 'r_opto'].to_numpy())

# Get time axis
time_ax = jpecc_df['time'].mean()

# Calculate asymmetry
asy_M2_mPFC = np.empty((M2_mPFC.shape[2], M2_mPFC.shape[0] - (asy_tb*2-1)))
for j in range(M2_mPFC.shape[2]):
    for i, k in enumerate(range(asy_tb, M2_mPFC.shape[0] - (asy_tb - 1))):
        M2_mPFC_slice = M2_mPFC[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_M2_mPFC[j, i] = (np.mean(M2_mPFC_slice[np.triu_indices(M2_mPFC_slice.shape[0], k=1)])
                             - np.mean(M2_mPFC_slice[np.tril_indices(M2_mPFC_slice.shape[0], k=-1)]))

asy_M2_ORB = np.empty((M2_ORB.shape[2], M2_ORB.shape[0] - (asy_tb*2-1)))
for j in range(M2_ORB.shape[2]):
    for i, k in enumerate(range(asy_tb, M2_ORB.shape[0] - (asy_tb - 1))):
        M2_ORB_slice = M2_ORB[k - asy_tb : i + asy_tb, k - asy_tb : k + asy_tb, j]
        asy_M2_ORB[j, i] = (np.mean(M2_ORB_slice[np.triu_indices(M2_ORB_slice.shape[0], k=1)])
                            - np.mean(M2_ORB_slice[np.tril_indices(M2_ORB_slice.shape[0], k=-1)]))
asy_time = jpecc_df['time'].mean()[range(asy_tb, M2_ORB.shape[0] - (asy_tb - 1))]

# Take diagonal lines (collapse over both time axes and take average)
# Trigger warning: this is extremely ugly code
diag_df = pd.DataFrame()
for j in range(M2_mPFC.shape[2]):
    for i in np.arange(-DIAGONALS, DIAGONALS+1):
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(M2_mPFC[:,:,j], offset=i),
            'diag_bl': (np.diagonal(M2_mPFC[:,:,j], offset=i)
                        - np.median(np.diagonal(M2_mPFC[:,:,j], offset=i)[time_ax[np.abs(i):] < 0])),
            'subject': j,
            'region_pair': 'M2-mPFC',
            'time': time_ax[np.abs(i):]})))
        if i == 0:
            this_time = time_ax
        else:
            this_time = time_ax[:-np.abs(i)]
        this_times = time_ax[:-np.abs(i)]
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(M2_mPFC[:,:,j], offset=i),
            'diag_bl': (np.diagonal(M2_mPFC[:,:,j], offset=i)
                        - np.median(np.diagonal(M2_mPFC[:,:,j], offset=i)[this_time < 0])),
            'subject': j,
            'region_pair': 'M2-mPFC',
            'time': this_time})))
for j in range(M2_ORB.shape[2]):
    for i in np.arange(-DIAGONALS, DIAGONALS+1):
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(M2_ORB[:,:,j], offset=i),
            'diag_bl': (np.diagonal(M2_ORB[:,:,j], offset=i)
                        - np.median(np.diagonal(M2_ORB[:,:,j], offset=i)[time_ax[np.abs(i):] < 0])),
            'subject': j,
            'region_pair': 'M2-ORB',
            'time': time_ax[np.abs(i):]})))
        if i == 0:
            this_time = time_ax
        else:
            this_time = time_ax[:-np.abs(i)]
        this_times = time_ax[:-np.abs(i)]
        diag_df = pd.concat((diag_df, pd.DataFrame(data={
            'diag': np.diagonal(M2_ORB[:,:,j], offset=i),
            'diag_bl': (np.diagonal(M2_ORB[:,:,j], offset=i)
                        - np.median(np.diagonal(M2_ORB[:,:,j], offset=i)[this_time < 0])),
            'subject': j,
            'region_pair': 'M2-ORB',
            'time': this_time})))
diag_df = diag_df.groupby(['time', 'subject', 'region_pair']).mean()
diag_df = diag_df.melt(ignore_index=False).reset_index()


# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax_cb) = plt.subplots(1, 3, figsize=(3.5, 1.75),
                                    gridspec_kw={'width_ratios': [1, 1, 0.2]}, dpi=dpi)

ax1.imshow(np.flipud(np.mean(M2_mPFC, axis=2)), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax1.invert_xaxis()
ax1.plot([0, 0], [-1, 3], color='white')
ax1.plot([-1, 3], [0, 0], color='white')
ax1.plot([-1, 3], [-1, 3], color='white')
ax1.set(ylabel='M2', xlabel='mPFC, time from stim. (s)', title='jPECC: M2 vs mPFC',
        xlim=[-1, 3], ylim=[-1, 3])

ax2.imshow(np.flipud(np.mean(M2_ORB, axis=2)), vmin=-0.5, vmax=0.5, cmap='icefire',
           extent=[time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2,
                   time_ax[0] - np.mean(np.diff(time_ax))/2, time_ax[-1] + np.mean(np.diff(time_ax))/2])
ax2.invert_xaxis()
ax2.plot([0, 0], [-1, 3], color='white')
ax2.plot([-1, 3], [0, 0], color='white')
ax2.plot([-1, 3], [-1, 3], color='white')
ax2.set(ylabel='M2', xlabel='mPFC, time from stim. (s)', title='M2 vs ORB',
        xlim=[-1, 3], ylim=[-1, 3])

ax_cb.axis('off')
plt.tight_layout()

cb_ax = f.add_axes([0.8, 0.3, 0.01, 0.5])
cbar = f.colorbar(mappable=ax2.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Population correlation (r)', rotation=270, labelpad=10)

plt.savefig(join(fig_path, 'jPECC_front.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, -1), 1, 3, color='royalblue', alpha=0.25, lw=0))
ax1.plot([-1, 3], [0, 0], ls='--', color='grey')
ax1.plot(asy_time, np.mean(asy_M2_mPFC, axis=0), color=colors['M2-mPFC'], label='M2-mPFC')
ax1.fill_between(asy_time,
                 np.mean(asy_M2_mPFC, axis=0)-(np.std(asy_M2_mPFC, axis=0)/np.sqrt(asy_M2_mPFC.shape[0])),
                 np.mean(asy_M2_mPFC, axis=0)+(np.std(asy_M2_mPFC, axis=0)/np.sqrt(asy_M2_mPFC.shape[0])),
                 alpha=0.2, color=colors['M2-mPFC'])
ax1.plot(asy_time, np.mean(asy_M2_ORB, axis=0), color=colors['M2-ORB'], label='M2-ORB')
ax1.fill_between(asy_time,
                 np.mean(asy_M2_ORB, axis=0)-(np.std(asy_M2_ORB, axis=0)/np.sqrt(asy_M2_ORB.shape[0])),
                 np.mean(asy_M2_ORB, axis=0)+(np.std(asy_M2_ORB, axis=0)/np.sqrt(asy_M2_ORB.shape[0])),
                 alpha=0.2, color=colors['M2-ORB'])
leg = ax1.legend(prop={'size': 5}, frameon=True, loc='lower left')
leg.get_frame().set_linewidth(0)
ax1.set(ylabel='jPECC asymmetry', xlabel='Time from stim. onset (s)', xticks=[-1, 0, 1, 2, 3],
        ylim=[-0.4, 0.3], xlim=[-1, 3], yticks=np.arange(-0.4, 0.31, 0.1))

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'jPECC_asymmetry.pdf'))

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.add_patch(Rectangle((0, 0), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag'], hue='region_pair', ax=ax1, ci=68,
             hue_order=['M2-ORB', 'M2-mPFC'], palette=[colors['M2-ORB'], colors['M2-mPFC']])
ax1.set(xlabel='Time (s)', ylabel='Canonical correlation (r)', xlim=[-1, 3], ylim=[0, 0.6],
        yticks=np.arange(0, 0.61, 0.1), xticks=[-1, 0, 1, 2, 3])
leg = ax1.legend(prop={'size': 5}, loc='upper left')
leg.get_frame().set_linewidth(0)

ax2.plot([-1, 3], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.3), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=diag_df[diag_df['variable'] == 'diag_bl'], hue='region_pair', ax=ax2, ci=68,
             hue_order=['M2-ORB', 'M2-mPFC'], palette=[colors['M2-ORB'], colors['M2-mPFC']])
ax2.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3], ylim=[-0.3, 0.3],
        yticks=np.arange(-0.3, 0.31, 0.1), xticks=[-1, 0, 1, 2, 3])
leg = ax2.legend(frameon=True, prop={'size': 5}, loc='upper left')
leg.get_frame().set_linewidth(0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA.pdf'))

