#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:12:54 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from serotonin_functions import paths, load_subjects, figure_style

# Settings
WIN_SIZE = 0.05
MIN_SUBJECTS = 2
N_MODES = 3
YLIM = 0.3

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA', 'StimOn')

# Load in data
cca_df = pd.read_pickle(join(save_path, f'cca_stimon_results_{WIN_SIZE}_binsize.pkl'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
cca_df = cca_df[cca_df['expression'] == 1].reset_index(drop=True)

# Restructue data into array per CCA mode
cca_opto, cca_no_opto = dict(), dict()
for i, region_pair in enumerate(cca_df['region_pair'].unique()):
    this_cca_opto = cca_df.loc[cca_df['region_pair'] == region_pair, 'r_baseline_opto'].to_numpy()
    this_cca_no_opto = cca_df.loc[cca_df['region_pair'] == region_pair, 'r_baseline_no_opto'].to_numpy()
    if len(this_cca_opto) >= MIN_SUBJECTS:
        this_mode_opto, this_mode_no_opto = [], []
        for j in range(len(this_cca_opto)):  # subjects
            if j == 0:
                this_mode_opto = this_cca_opto[j].copy()
                this_mode_no_opto = this_cca_no_opto[j].copy()
            else:
                this_mode_opto = np.dstack((this_mode_opto, this_cca_opto[j]))
                this_mode_no_opto = np.dstack((this_mode_no_opto, this_cca_no_opto[j]))
        cca_opto[region_pair] = this_mode_opto  # CCA modes X timebins X subjects
        cca_no_opto[region_pair] = this_mode_no_opto  # CCA modes X timebins X subjects

#  Plot
time_ax = cca_df['time'][0]
colors, dpi = figure_style()
for i, region_pair in enumerate(cca_opto.keys()):
    f, axs = plt.subplots(1, N_MODES, figsize=(1.75*N_MODES, 2), dpi=dpi)
    for mm in range(N_MODES):
        # Laser on
        this_mean = np.mean(cca_opto[region_pair][mm, :, :], axis=1)
        this_err = (np.std(cca_opto[region_pair][mm, :, :], axis=1)
                    / np.sqrt(cca_opto[region_pair].shape[2]))
        axs[mm].plot(time_ax, this_mean, color=colors['stim'])
        axs[mm].fill_between(time_ax, this_mean - this_err, this_mean + this_err, alpha=0.5,
                             color=colors['stim'])

        # Laser off
        this_mean = np.mean(cca_no_opto[region_pair][mm, :, :], axis=1)
        this_err = (np.std(cca_no_opto[region_pair][mm, :, :], axis=1)
                    / np.sqrt(cca_no_opto[region_pair].shape[2]))
        axs[mm].plot(time_ax, this_mean, color=colors['no-stim'])
        axs[mm].fill_between(time_ax, this_mean - this_err, this_mean + this_err, alpha=0.5,
                             color=colors['no-stim'])

        axs[mm].set(xlabel='Time from visual stim. onset (s)', ylabel='Canonical correlation (r)',
                    title=f'Mode {mm+1}', xticks=[-1, 0, 1, 2, 3], ylim=[-YLIM, YLIM],
                    yticks=np.arange(-YLIM, YLIM+0.01, 0.1), xlim=[-1, 2])
    f.suptitle(f'{region_pair}')
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(fig_path, f'{region_pair}.pdf'))
    plt.close(f)