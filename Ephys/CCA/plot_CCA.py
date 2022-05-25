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

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA', 'RegionPairs')

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results.csv'))

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    cca_df.loc[cca_df['subject'] == nickname, 'expression'] = subjects.loc[subjects['subject'] == nickname, 'expression'].values[0]
cca_df = cca_df[cca_df['expression'] == 1]

# Get slices of dataframe during stim and baseline
cca_stim = cca_df[(cca_df['time'] < 1) & (cca_df['time'] > 0)]
cca_baseline = cca_df[(cca_df['time'] < 1) & (cca_df['time'] > 0)]

# Get averages
cca_merged_stim = cca_stim.groupby(['region_1', 'region_2']).mean()
cca_merged_stim = cca_baseline.groupby(['region_1', 'region_2']).mean()

# %% Plot
for i, region_pair in enumerate(cca_df['region_pair'].unique()):
    colors, dpi = figure_style()
    f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
    sns.lineplot(x='time', y='r_baseline', data=cca_df[cca_df['region_pair'] == region_pair], ax=ax1,
                 palette='tab10', ci=68)
    ax1.plot(ax1.get_xlim(), [0, 0], ls='--', color='grey')
    ax1.set(title=f'{region_pair}', ylabel='Population correlation (r)', xticks=[-1, 0, 1, 2, 3])
    
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(fig_path, f'{region_pair}.jpg'), dpi=300)
    plt.close(f)