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

# Settings
REGIONS = ['PL', 'ORBl', 'IL']

# Load in data
_, fig_path, save_path = paths()
pop_df = pd.read_csv(join(save_path, 'pop_mean_var.csv'))

# Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi)
#sns.lineplot(x='time', y='mean', data=pop_df[pop_df['region'].isin(REGIONS)], hue='region', ax=ax1)
#sns.lineplot(x='time', y='var', data=pop_df[pop_df['region'].isin(REGIONS)], hue='region', ax=ax2)
sns.lineplot(x='time', y='mean_bl', data=pop_df, hue='region', ax=ax1)
sns.lineplot(x='time', y='std_bl', data=pop_df, hue='region', ax=ax2)

