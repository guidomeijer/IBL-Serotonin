# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:36:23 2022

@author: Guido
"""

from os.path import join
from serotonin_functions import paths, figure_style
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load in data
_, fig_path, save_path = paths()
pop_df = pd.read_csv(join(save_path, 'population_metrics.csv'))
mua_df = pd.read_csv(join(save_path, 'multi_unit_activity.csv'))

# Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi)
sns.lineplot(x='time', y='mean_bl', data=pop_df, hue='region', ax=ax1)
sns.lineplot(x='time', y='std_bl', data=pop_df, hue='region', ax=ax2)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi)
sns.lineplot(x='time', y='sparsity_bl', data=pop_df, hue='region', ax=ax1)
sns.lineplot(x='time', y='noise_corr_bl', data=pop_df, hue='region', ax=ax2)

