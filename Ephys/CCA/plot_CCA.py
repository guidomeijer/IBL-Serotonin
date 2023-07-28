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
from stim_functions import paths, load_subjects, figure_style
from dlc_functions import smooth_interpolate_signal_sg

# Settings

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Load in data
cca_df = pd.read_pickle(join(save_path, 'cca_results.pickle'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
cca_df = cca_df[cca_df['sert-cre'] == 1]

# Get time axis
time_ax = np.round(cca_df['time'].mean(), 3)

# %%
colors, dpi = figure_style()
for i in cca_df.index:
    f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
    ax1.fill_between(cca_df.loc[i, 'time'], cca_df.loc[i, 'r_mean'] - cca_df.loc[i, 'r_std'],
                     cca_df.loc[i, 'r_mean'] + cca_df.loc[i, 'r_std'], alpha=0.2)
    ax1.plot(cca_df.loc[i, 'time'], cca_df.loc[i, 'r_mean'])
    ax1.set(ylabel='Canonical correlation (r)', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3],
            title=f'{cca_df.loc[i, "subject"]} {cca_df.loc[i, "region_1"]}-{cca_df.loc[i, "region_2"]}')

    sns.despine(trim=True)
    plt.tight_layout()