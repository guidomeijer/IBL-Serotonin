#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:00:47 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from serotonin_functions import figure_style, paths, load_subjects
from os.path import join

# Get paths
fig_path, save_path = paths()

# Load in data
state_trans_df = pd.read_csv(join(save_path, 'HMM', 'all_state_trans.csv'))
p_state_df = pd.read_csv(join(save_path, 'HMM', 'p_state.csv'))

# Average over mice first
state_trans_df = state_trans_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()
p_state_df = p_state_df.groupby(['subject', 'time', 'region']).mean(numeric_only=True).reset_index()

# Only select sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    state_trans_df.loc[state_trans_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
    p_state_df.loc[p_state_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
state_trans_df = state_trans_df[state_trans_df['sert-cre'] == 1]
p_state_df = p_state_df[p_state_df['sert-cre'] == 1]

# %% Plot
colors, dpi = figure_style()

f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -4), 1, 5, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=state_trans_df[state_trans_df['region'] == region], x='time', y='trans_rate_bl',
                 color='k', errorbar='se', ax=axs[i])
    axs[i].set(ylabel='State change rate', xlabel='Time (s)', title=region, ylim=[-4, 1],
               yticks=[-4, -3, -2, -1, 0, 1], xticks=[-1, 0, 1, 2, 3, 4])
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)

# %%
f, axs = plt.subplots(2, 4, figsize=(7, 3.5), dpi=dpi)
axs = np.concatenate(axs)
for i, region in enumerate(np.unique(p_state_df['region'])):

    axs[i].add_patch(Rectangle((0, -0.3), 1, 0.6, color='royalblue', alpha=0.25, lw=0))
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='state_incr',
                 color=colors['enhanced'], errorbar='se', ax=axs[i])
    sns.lineplot(data=p_state_df[p_state_df['region'] == region], x='time', y='state_decr',
                 color=colors['suppressed'], errorbar='se', ax=axs[i])
    axs[i].set(ylabel='P(state)', xlabel='Time (s)', title=region, ylim=[-0.2, 0.2],
               yticks=[-0.2, 0, 0.2], xticks=[-1, 0, 1, 2, 3, 4])
axs[-1].axis('off')
plt.tight_layout()
sns.despine(trim=True)