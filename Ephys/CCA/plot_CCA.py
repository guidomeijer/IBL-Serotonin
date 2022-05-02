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

fig_path, save_path = paths()

# Load in data
cca_df = pd.read_csv(join(save_path, 'cca_results_all.csv'))

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
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)

