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
from dlc_functions import smooth_interpolate_signal_sg

# Settings
REGION_PAIRS = ['M2-mPFC', 'M2-OFC']
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

        # Smooth signal
        smooth_cca_pre = smooth_interpolate_signal_sg(pre_cca[rp][:,jj], window=11)
        smooth_cca_post = smooth_interpolate_signal_sg(post_cca[rp][:,jj], window=11)

        # Add to dataframe
        cca_df = pd.concat((cca_df, pd.DataFrame(data={
            'pre_cca': smooth_cca_pre, 'post_cca': smooth_cca_post,
            'pre_cca_bl': smooth_cca_pre - np.median(smooth_cca_pre[time_ax < 0]),
            'post_cca_bl': smooth_cca_post - np.median(smooth_cca_post[time_ax < 0]),
            'subject': subject, 'region_pair': rp, 'time': time_ax})))

# Take average per timepoint
cca_df = cca_df.groupby(['time', 'subject', 'region_pair']).mean()
cca_long_df = cca_df.melt(ignore_index=False).reset_index()
cca_df = cca_df.reset_index()


# %%
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot([-1, 3], [0, 0], ls='--', color='grey')
ax1.add_patch(Rectangle((0, -0.6), 1, 1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'pre_cca_bl'],
             hue='region_pair', ax=ax1, ci=68, hue_order=['M2-mPFC', 'M2-OFC'],
             palette=[colors['mPFC'], colors['OFC']])
ax1.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3],
        ylim=[-0.6, 0.4], yticks=np.arange(-0.6, 0.41, 0.2), xticks=[-1, 0, 1, 2, 3],
        title='CCA axis fit to [-1s, 0s]')
leg_handles, _ = ax1.get_legend_handles_labels()
leg_labels = [f'M2-mPFC (n={cca_df[cca_df["region_pair"] == "M2-mPFC"]["subject"].unique().size})',
              f'M2-OFC (n={cca_df[cca_df["region_pair"] == "M2-OFC"]["subject"].unique().size})']
leg = ax1.legend(leg_handles, leg_labels, prop={'size': 5}, bbox_to_anchor=(0.4, 0.82))
leg.get_frame().set_linewidth(0)

ax2.plot([-1, 3], [0, 0], ls='--', color='grey')
ax2.add_patch(Rectangle((0, -0.6), 1, 1, color='royalblue', alpha=0.25, lw=0))
sns.lineplot(x='time', y='value', data=cca_long_df[cca_long_df['variable'] == 'post_cca_bl'],
             hue='region_pair', ax=ax2, ci=68, hue_order=['M2-mPFC', 'M2-OFC'],
             palette=[colors['mPFC'], colors['OFC']], legend=None)
ax2.set(xlabel='Time (s)', ylabel='Canonical correlation \n over baseline (r)', xlim=[-1, 3],
        ylim=[-0.6, 0.4], yticks=np.arange(-0.6, 0.41, 0.2), xticks=[-1, 0, 1, 2, 3],
        title='CCA axis fit to [0s, 1s]')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'CCA_pre_post_M2_mPFC_OFC.pdf'))

# %% Per mouse
mice = cca_long_df['subject'].unique()
f, axs = plt.subplots(1, mice.shape[0], figsize=(1.75*mice.shape[0], 1.75), dpi=dpi)
for i, mouse in enumerate(mice):
    cca_slice_df = cca_long_df[(cca_long_df['subject'] == mouse)  & (cca_long_df['variable'] == 'pre_cca_bl')]
    axs[i].plot([-1, 3], [0, 0], ls='--', color='grey')
    axs[i].add_patch(Rectangle((0, -1), 1, 1.4, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(x='time', y='value', data=cca_slice_df, hue='region_pair', ax=axs[i],
                 palette=[colors['mPFC'], colors['OFC']], hue_order=['M2-mPFC', 'M2-OFC'],
                 legend=None)
    axs[i].set(xlabel='Time (s)', xlim=[-1, 3], ylim=[-1, 0.4], yticks=np.arange(-1, 0.41, 0.2),
               xticks=[-1, 0, 1, 2, 3], title=f'{mouse}')
    if i == 0:
        axs[i].set(ylabel='Canonical correlation \n over baseline (r)')
    else:
        axs[i].set(ylabel='')

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'CCA_pre_M2_mPFC_OFC_per_mouse.pdf'))

