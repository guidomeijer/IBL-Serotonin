#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:01:51 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from serotonin_functions import (load_passive_opto_times, get_neuron_qc, paths, query_ephys_sessions,
                                 figure_style, load_subjects, remap)

# Get paths
fig_path, save_path = paths()

# Load in data
up_down_df = pd.read_csv(join(save_path, 'up_down_states.csv'))

# Get binsize
bin_size = np.round(up_down_df['time'].values[1] - up_down_df['time'].values[0], 1)

# Get state feature data
state_feat_df = pd.DataFrame()
for i, subject in enumerate(np.unique(up_down_df['subject'])):
    for k, region in enumerate(np.unique(up_down_df['region'])):

        # Get pre-opto state features
        state = up_down_df.loc[(up_down_df['region'] == region) & (up_down_df['opto'] == 0)
                               & (up_down_df['subject'] == subject), 'state'].values
        state_changes = np.concatenate((np.zeros(1), np.diff(state)))
        change_inds = np.where(state_changes != 0)[0]

        for j, ch_ind in enumerate(change_inds[:-1]):

            # Get state features
            this_state = state[ch_ind]
            state_dur = (change_inds[j+1] - change_inds[j]) * bin_size

            # Add to df
            state_feat_df = pd.concat((state_feat_df, pd.DataFrame(index=[state_feat_df.shape[0]+1], data={
                'subject': subject, 'region': region, 'state': this_state, 'state_dur': state_dur,
                'opto': 0})))

        # Get opto state features
        state = up_down_df.loc[(up_down_df['region'] == region) & (up_down_df['opto'] == 1)
                               & (up_down_df['subject'] == subject), 'state'].values
        state_changes = np.concatenate((np.zeros(1), np.diff(state)))
        change_inds = np.where(state_changes != 0)[0]

        for j, ch_ind in enumerate(change_inds[:-1]):

            # Get state features
            this_state = state[ch_ind]
            state_dur = (change_inds[j+1] - change_inds[j]) * bin_size

            # Add to df
            state_feat_df = pd.concat((state_feat_df, pd.DataFrame(index=[state_feat_df.shape[0]+1], data={
                'subject': subject, 'region': region, 'state': this_state, 'state_dur': state_dur,
                'opto': 1})))

# Do some data cleaning
#state_feat_df = state_feat_df[(state_feat_df['state_dur'] > 0.6) & (state_feat_df['state_dur'] < 4)]
#state_feat_df = state_feat_df[(state_feat_df['state_dur'] > 0.6) & (state_feat_df['state_dur'] < 4)]

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)

sns.histplot(x='state_dur', hue='opto', ax=ax1, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['state'] == 0)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax1.set(xlim=[0, 4], xlabel='Down state duration', title='Cortex')

sns.histplot(x='state_dur', hue='opto', ax=ax2, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['state'] == 1)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax2.set(xlim=[0, 4], xlabel='Up state duration')

sns.histplot(x='state_dur', hue='opto', ax=ax3, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'CNU') & (state_feat_df['state'] == 0)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax3.set(xlim=[0, 4], xlabel='Down state duration', title='Striatum')

sns.histplot(x='state_dur', hue='opto', ax=ax4, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'CNU') & (state_feat_df['state'] == 1)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax4.set(xlim=[0, 4], xlabel='Up state duration')

sns.despine(trim=True)
plt.tight_layout()

# %%
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'Isocortex'],
            ax=ax1, palette=[colors['stim'], colors['no-stim']], hue_order=[1, 0], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Isocortex') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax2.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax2.text(1, 4, '*', ha='center', va='center')
ax1.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Cortex', ylabel='State duration (s)',
        ylim=[0, 6])
ax1.legend().set_visible(False)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'CNU'],
            ax=ax2, palette=[colors['stim'], colors['no-stim']], hue_order=[1, 0], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'CNU') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'CNU') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'CNU') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'CNU') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax2.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax2.text(1, 4, '*', ha='center', va='center')
ax2.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Striatum', ylabel='State duration (s)',
        ylim=[0, 6])
ax2.legend().set_visible(False)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'TH'],
            ax=ax3, palette=[colors['stim'], colors['no-stim']], hue_order=[1, 0], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'TH') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'TH') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'TH') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'TH') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax3.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax3.text(1, 4, '*', ha='center', va='center')
ax3.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Thalamus', ylabel='State duration (s)',
        ylim=[0, 6])
ax3.legend().set_visible(False)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'CTXsp'],
            ax=ax4, palette=[colors['stim'], colors['no-stim']], hue_order=[1, 0], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'CTXsp') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'CTXsp') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'CTXsp') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'CTXsp') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax4.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax4.text(1, 4, '*', ha='center', va='center')
ax4.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Amygdala', ylabel='State duration (s)',
        ylim=[0, 6])
ax4.legend().set_visible(False)

sns.despine(trim=True)
plt.tight_layout()