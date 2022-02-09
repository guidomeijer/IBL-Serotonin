#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from serotonin_functions import paths, figure_style, load_subjects

# Settings
MIN_NEURONS = 5
MIN_FR = 0.01  # spks/s
BASELINE = [-0.5, 0]
STIM = [0, 0.5]
PLOT = True
NEURON_QC = True
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'Correlations')

# Load in data
corr_df = pd.read_csv(join(save_path, 'correlations_late.csv'))

# Only sert-cre mice
subjects = load_subjects()
corr_df = corr_df[corr_df['subject'].isin(
    subjects.loc[subjects['sert-cre'] == 1, 'subject'].values)]

# Only keep thalamus
thalamus_df = corr_df[corr_df['region'].str.contains('Thalamus')]
for i, ind in enumerate(thalamus_df.index.values):
    thalamus_df.loc[ind, 'region'] = thalamus_df.loc[ind, 'region'][10:-1]

# %%
colors, dpi = figure_style()

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
sns.barplot(x='region', y='corr_change', data=thalamus_df, color='orange', ci=0, ax=ax1)
sns.swarmplot(x='region', y='corr_change', data=thalamus_df, color='grey', ax=ax1)
ax1.plot([-1, ax1.get_xlim()[1]], [0, 0], color='k')
ax1.margins(x=0)
ax1.set(ylabel='Pairwise correlation stim-baseline (r)', xlabel='Thalamic nuclei')

plt.tight_layout()
sns.despine(trim=True)


f, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)
sns.barplot(x='region', y='corr_change', data=corr_df, color='orange', ci=0, ax=ax1)
sns.swarmplot(x='region', y='corr_change', data=corr_df, color='grey', ax=ax1)
ax1.plot([-1, ax1.get_xlim()[1]], [0, 0], color='k')
ax1.margins(x=0)
ax1.set(ylabel='Pairwise correlation stim-baseline (r)', xlabel='Thalamic nuclei')
plt.xticks(rotation=90)

plt.tight_layout()
sns.despine(trim=True)
