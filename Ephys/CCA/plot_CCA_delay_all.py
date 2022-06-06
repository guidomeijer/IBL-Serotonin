

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
PER_SUBJECT = False
ASYM_TIME = 0.1
CCA_TIME = 0
BIN_SIZE = 0.05
TIME_WIN = 0.5
MIN_N = 2

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Load in data
jpecc_df = pd.read_pickle(join(save_path, f'jPECC_delay_{BIN_SIZE}_binsize.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    jpecc_df.loc[jpecc_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
jpecc_df = jpecc_df[jpecc_df['sert-cre'] == 1]

# Get time axis
time_ax = np.round(jpecc_df['time'].mean(), 3)
time_asy = np.round(jpecc_df['delta_time'].mean(), 3)

# Get 3D array of all jPECC
jPECC, asym, cca_df = dict(), dict(), pd.DataFrame()
for i, rp in enumerate(np.unique(jpecc_df['region_pair'])):

    jPECC[rp] = np.dstack(jpecc_df.loc[jpecc_df['region_pair'] == rp, 'r_opto'].to_numpy())

    for jj, subject in zip(range(jPECC[rp].shape[2]), jpecc_df.loc[jpecc_df['region_pair'] == rp, 'subject']):

        # Do some smoothing
        jPECC[rp][:, :, jj] = gaussian_filter(jPECC[rp][:, :, jj], 1)

        # Get max CCA during and after stim
        this_cca = np.squeeze(np.median(jPECC[rp][:, (time_asy >= -CCA_TIME)
                                                  & (time_asy <= CCA_TIME), jj], axis=1))

        # Add to dataframe
        cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0] + 1], data={
            'cca_bl': np.median(this_cca[(time_ax > -TIME_WIN) & (time_ax < 0)]),
            'cca_stim': np.median(this_cca[(time_ax > 0) & (time_ax < TIME_WIN)]),
            'cca_after_stim': np.median(this_cca[(time_ax > 1) & (time_ax < 1 + TIME_WIN)]),
            'subject': subject, 'region_pair': rp})))

# Baseline subtract
cca_df['cca_stim'] = cca_df['cca_stim'] - cca_df['cca_bl']
cca_df['cca_after_stim'] = cca_df['cca_after_stim'] - cca_df['cca_bl']

cca_grouped_df = cca_df.groupby('region_pair').mean()
cca_grouped_df['n'] = cca_df.groupby('region_pair').size()
cca_grouped_df = cca_grouped_df.reset_index()
for i, rp in enumerate(cca_grouped_df['region_pair']):
    cca_grouped_df.loc[i, 'region_1'] = rp.split('-')[0]
    cca_grouped_df.loc[i, 'region_2'] = rp.split('-')[1]

# Apply selection
cca_grouped_df = cca_grouped_df[cca_grouped_df['n'] >= MIN_N]

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2.5), dpi=dpi)
sns.barplot(x='region_pair', y='cca_bl', data=cca_grouped_df, ax=ax1,
            order=cca_grouped_df.sort_values('cca_bl', ascending=False)['region_pair'],
            color=colors['general'])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set(ylabel='Baseline canonical correlation (r)', xlabel='')

sns.barplot(x='region_pair', y='cca_stim', data=cca_grouped_df, ax=ax2,
            order=cca_grouped_df.sort_values('cca_stim', ascending=False)['region_pair'],
            color=colors['general'])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set(ylabel='Canonical correlation change \n during stim. (r)', xlabel='')

sns.barplot(x='region_pair', y='cca_after_stim', data=cca_grouped_df, ax=ax3,
            order=cca_grouped_df.sort_values('cca_after_stim', ascending=False)['region_pair'],
            color=colors['general'])
ax3.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax3.set(ylabel='Canonical correlation change \n after stim. (r)', xlabel='')

plt.tight_layout()
sns.despine(trim=True)

#plt.savefig(join(fig_path, 'jPECC_PPC_Hipp_Thal.pdf'))

