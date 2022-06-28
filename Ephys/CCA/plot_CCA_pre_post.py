# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:21:26 2022

@author: Guido
"""

import pandas as pd
import numpy as np
from os.path import join
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from serotonin_functions import paths, load_subjects, figure_style

# Settings
REGION_PAIRS = ['M2-mPFC']
PER_SUBJECT = False

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Load in data
load_cca_df = pd.read_pickle(join(save_path, 'CCA_pre_post_opto.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    load_cca_df.loc[load_cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
load_cca_df = load_cca_df[load_cca_df['sert-cre'] == 1]

# Get time axis
time_ax = np.round(load_cca_df['time'].mean(), 3)

# Get 3D array of all jPECC
pre_cca, post_cca, cca_df = dict(), dict(), pd.DataFrame()
for i, rp in enumerate(REGION_PAIRS):

    # Load in data from dataframe
    this_cca = np.dstack(load_cca_df.loc[load_cca_df['region_pair'] == rp, 'r_pre'].to_numpy())
    pre_cca[rp] = this_cca[:, 0, :]  # Take first CCA mode
    this_cca = np.dstack(load_cca_df.loc[load_cca_df['region_pair'] == rp, 'r_post'].to_numpy())
    post_cca[rp] = this_cca[:, 0, :]  # Take first CCA mode

    for jj, subject in zip(range(pre_cca[rp].shape[1]), load_cca_df.loc[load_cca_df['region_pair'] == rp, 'subject']):

        # Add to dataframe
        cca_df = pd.concat((cca_df, pd.DataFrame(data={
            'pre_cca': pre_cca[rp][:,jj], 'post_cca': post_cca[rp][:,jj],
            'pre_cca_bl': pre_cca[rp][:,jj] - np.median(pre_cca[rp][time_ax < 0, jj]),
            'post_cca_bl': post_cca[rp][:,jj] - np.median(post_cca[rp][time_ax < 0, jj]),
            'subject': subject, 'region_pair': rp, 'time': time_ax})))

# Take average per timepoint
cca_df = cca_df.groupby(['time', 'subject', 'region_pair']).mean()
cca_long_df = cca_df.melt(ignore_index=False).reset_index()
cca_df = cca_df.reset_index()


# %%
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot([-1, 3], [0, 0], ls='--', color='grey')
ax1.add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=cca_long_df, hue='variable', ax=ax1, ci=68,
             hue_order=['pre_cca_bl', 'post_cca_bl'])
#ax1.set(xlim=[-0.5, 1])

# %%

ax1.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3], ylim=[-0.4, 0.4],
        yticks=np.arange(-0.4, 0.41, 0.2), xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = [f'M2-mPFC (n={cca_df[cca_df["region_pair"] == "M2-mPFC"]["subject"].unique().size})',
              f'M2-OFC (n={cca_df[cca_df["region_pair"] == "M2-ORB"]["subject"].unique().size})']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=(0.4, 0.82))
leg.get_frame().set_linewidth(0)

ax2.plot([-1, 3], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
if PER_SUBJECT:
    sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'cca_bl'], hue='region_pair', ax=ax2,
                 estimator=None, units='subject',
                 hue_order=['ORB-mPFC'], palette=[colors['mPFC']])
else:
    sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'cca_bl'], hue='region_pair', ax=ax2, ci=68,
                 hue_order=['ORB-mPFC'], palette=[colors['M2']])
ax2.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3], ylim=[-0.4, 0.4],
        yticks=np.arange(-0.4, 0.41, 0.2), xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax2.get_legend_handles_labels()
leg_labels = [f'OFC-mPFC (n={cca_df[cca_df["region_pair"] == "ORB-mPFC"]["subject"].unique().size})']
leg = ax2.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left')
leg.get_frame().set_linewidth(0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA_M2_mPFC_ORB.pdf'))

# %% Per mouse
mice = cca_long_df['subject'].unique()
f, axs = plt.subplots(1, mice.shape[0], figsize=(1.75*mice.shape[0], 1.75), dpi=dpi)
for i, mouse in enumerate(mice):
    cca_slice_df = cca_long_df[(cca_long_df['subject'] == mouse)  & (cca_long_df['variable'] == 'cca_bl')]
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(x='time', y='value', data=cca_slice_df, hue='region_pair', ax=axs[i],
                 palette=[colors['mPFC'], colors['OFC']], hue_order=['M2-mPFC', 'M2-ORB'],
                 legend=None)
    axs[i].set(xlabel='Time (s)', xlim=[-1, 3], ylim=[-0.4, 0.4], yticks=np.arange(-0.4, 0.41, 0.2),
               xticks=[-1, 0, 1, 2, 3], title=f'{mouse}')
    if i == 0:
        axs[i].set(ylabel='Canonical correlation \n over baseline (r)')
    else:
        axs[i].set(ylabel='')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_CCA_M2_mPFC_OFC_per_mouse.pdf'))

# %%
YLIM = 0.4
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot([-1, 3], [0, 0], ls='--', color='grey')
ax1.add_patch(Rectangle((0, -YLIM), 1, YLIM*2, color='royalblue', alpha=0.25, lw=0))
if PER_SUBJECT:
    sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'asym'], hue='region_pair', ax=ax1,
                 estimator=None, units='subject', style='subject',
                 hue_order=['M2-mPFC', 'M2-ORB'], palette=[colors['mPFC'], colors['OFC']])
else:
    sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'asym'], hue='region_pair', ax=ax1, ci=68,
                 hue_order=['M2-mPFC', 'M2-ORB'], palette=[colors['mPFC'], colors['OFC']])
ax1.set(xlabel='Time (s)', ylabel=r'Asymmetry ($\bigtriangleup$r)', xlim=[-1, 3], ylim=[-YLIM, YLIM],
        yticks=np.arange(-YLIM, YLIM+0.01, 0.2), xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = ['M2-mPFC', 'M2-OFC']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left')
leg.get_frame().set_linewidth(0)

ax2.plot([-1, 3], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.2), 1, 0.4, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'asym'], hue='region_pair', ax=ax2, ci=68,
             hue_order=['ORB-mPFC'], palette=[colors['M2']])
ax2.set(xlabel='Time (s)', ylabel='Asymmetry', xlim=[-1, 3], ylim=[-YLIM, YLIM],
        yticks=np.arange(-YLIM, YLIM+0.01, 0.2), xticks=[-1, 0, 1, 2, 3])
leg_handles, _ = ax2.get_legend_handles_labels()
leg_labels = ['OFC-mPFC']
leg = ax2.legend(leg_handles, leg_labels, prop={'size': 5}, loc='lower left')
leg.get_frame().set_linewidth(0)

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_asymmetry_M2_mPFC_OFC.pdf'))

# %%
colors, dpi = figure_style()
mice = cca_long_df['subject'].unique()
f, axs = plt.subplots(1, mice.shape[0], figsize=(1.75*mice.shape[0], 1.75), dpi=dpi)
for i, mouse in enumerate(mice):
    cca_slice_df = cca_long_df[(cca_long_df['subject'] == mouse)  & (cca_long_df['variable'] == 'asym')]
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].add_patch(Rectangle((0, -0.4), 1, 0.8, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(x='time', y='value', data=cca_slice_df, hue='region_pair', ax=axs[i],
                 palette=[colors['mPFC'], colors['OFC']], hue_order=['M2-mPFC', 'M2-ORB'],
                 legend=None)
    axs[i].set(xlabel='Time (s)', xlim=[-1, 3], ylim=[-0.4, 0.4], yticks=np.arange(-0.4, 0.41, 0.2),
               xticks=[-1, 0, 1, 2, 3], title=f'{mouse}')
    if i == 0:
        axs[i].set(ylabel=r'Asymmetry ($\bigtriangleup$r)')
    else:
        axs[i].set(ylabel='')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'jPECC_asymmetry_M2_mPFC_OFC_per_mouse.pdf'))
