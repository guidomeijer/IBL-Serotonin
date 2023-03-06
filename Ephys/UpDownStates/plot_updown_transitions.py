#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:11:55 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from serotonin_functions import figure_style, paths
from os.path import join

# Get paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'UpDownStates')

# Load in data
anesthesia_df = pd.read_csv(join(save_path, 'updown_states_anesthesia.csv'))

# Average over mice first
anes_mean_df = anesthesia_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()

<<<<<<< Updated upstream
=======
# Plot awake
colors, dpi = figure_style()
all_regions = np.unique(awake_mean_df['region'])
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(all_regions):
    sns.lineplot(data=awake_mean_df[awake_mean_df['region'] == region], x='time', y='to_down_peths',
                 ax=axs[i], color=colors['suppressed'], errorbar='se')
    sns.lineplot(data=awake_mean_df[awake_mean_df['region'] == region], x='time', y='to_up_peths',
                 ax=axs[i], color=colors['enhanced'], errorbar='se')
    n_sub = len(np.unique(awake_mean_df.loc[awake_mean_df['region'] == region, 'subject']))
    axs[i].set(title=f'{region} n={n_sub} mice', ylabel='State transition rate', xlabel='Time (s)',
               ylim=[0, 0.4], xticks=[-1, 0, 1, 2, 3, 4])
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'updown_trans_awake.jpg'), dpi=600)

>>>>>>> Stashed changes
# Plot anesthesia
colors, dpi = figure_style()
all_regions = np.unique(anes_mean_df['region'])
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(all_regions):
    axs[i].plot([-1, 4], [0.5, 0.5], ls='--', color='grey')
    sns.lineplot(data=anes_mean_df[anes_mean_df['region'] == region], x='time', y='p_down',
                 ax=axs[i], color=colors['suppressed'], errorbar='se')
    n_sub = len(np.unique(anes_mean_df.loc[anes_mean_df['region'] == region, 'subject']))
    axs[i].set(title=f'{region} n={n_sub} mice', ylabel='P(down)', xlabel='Time (s)',
               ylim=[0, 1], xticks=[-1, 0, 1, 2, 3, 4])
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'updown_trans_anesthesia.jpg'), dpi=600)