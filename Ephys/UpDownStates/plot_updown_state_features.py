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

from statsmodels.stats.oneway import anova_oneway
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison


import matplotlib.pyplot as plt
from serotonin_functions import (load_passive_opto_times, get_neuron_qc, paths, query_ephys_sessions,
                                 figure_style, load_subjects, remap)

# Get paths
fig_path, save_path = paths()

# Load in data
up_down_df = pd.read_csv(join(save_path, 'updown_state_anesthesia.csv'))

# Get binsize
bin_size = np.round(up_down_df['time'].values[1] - up_down_df['time'].values[0], 1)

# Get state feature data
state_feat_df = pd.DataFrame()
for i, pid in enumerate(np.unique(up_down_df['pid'])):
    for k, region in enumerate(np.unique(up_down_df.loc[up_down_df['pid'] == pid, 'region'])):

        # Get subject
        subject = up_down_df.loc[up_down_df['pid'] == pid, 'subject'].values[0]

        # Get pre-opto state features
        state = up_down_df.loc[(up_down_df['region'] == region) & (up_down_df['opto'] == 0)
                               & (up_down_df['pid'] == pid), 'state'].values
        state_changes = np.concatenate((np.zeros(1), np.diff(state)))
        change_inds = np.where(state_changes != 0)[0]

        for j, ch_ind in enumerate(change_inds[:-1]):

            # Get state features
            this_state = state[ch_ind]
            state_dur = (change_inds[j+1] - change_inds[j]) * bin_size

            # Add to df
            state_feat_df = pd.concat((state_feat_df, pd.DataFrame(index=[state_feat_df.shape[0]+1], data={
                'subject': subject, 'region': region, 'state': this_state, 'state_dur': state_dur,
                'pid': pid, 'opto': 0})))

        # Get opto state features
        state = up_down_df.loc[(up_down_df['region'] == region) & (up_down_df['opto'] == 1)
                               & (up_down_df['pid'] == pid), 'state'].values
        state_changes = np.concatenate((np.zeros(1), np.diff(state)))
        change_inds = np.where(state_changes != 0)[0]

        for j, ch_ind in enumerate(change_inds[:-1]):

            # Get state features
            this_state = state[ch_ind]
            state_dur = (change_inds[j+1] - change_inds[j]) * bin_size

            # Add to df
            state_feat_df = pd.concat((state_feat_df, pd.DataFrame(index=[state_feat_df.shape[0]+1], data={
                'subject': subject, 'region': region, 'state': this_state, 'state_dur': state_dur,
                'pid': pid, 'opto': 1})))

# Do some data cleaning
state_feat_df = state_feat_df[state_feat_df['state_dur'] > 0.25]
#state_feat_df = state_feat_df[(state_feat_df['state_dur'] > 0.6) & (state_feat_df['state_dur'] < 4)]

# %% Plot

colors, dpi = figure_style()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)

sns.histplot(x='state_dur', hue='opto', ax=ax1, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Cortex') & (state_feat_df['state'] == 0)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax1.set(xlim=[0, 4], xlabel='Down state duration', title='Cortex')

sns.histplot(x='state_dur', hue='opto', ax=ax2, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Cortex') & (state_feat_df['state'] == 1)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax2.set(xlim=[0, 4], xlabel='Up state duration')

sns.histplot(x='state_dur', hue='opto', ax=ax3, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Striatum') & (state_feat_df['state'] == 0)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax3.set(xlim=[0, 4], xlabel='Down state duration', title='Striatum')

sns.histplot(x='state_dur', hue='opto', ax=ax4, element='step', palette=[colors['stim'], colors['no-stim']],
             data=state_feat_df[(state_feat_df['region'] == 'Striatum') & (state_feat_df['state'] == 1)],
             hue_order=[1, 0], stat="density", common_norm=False, legend=False)
ax4.set(xlim=[0, 4], xlabel='Up state duration')

sns.despine(trim=True)
plt.tight_layout()

# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'Cortex'],
            ax=ax1, palette=[colors['no-stim'], colors['stim']], hue_order=[0, 1], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Cortex') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Cortex') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Cortex') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Cortex') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax1.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax1.text(1, 4, '*', ha='center', va='center', fontsize=12)
ax1.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Cortex', ylabel='State duration (s)',
        ylim=[0, 6])
ax1.legend().set_visible(False)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'Striatum'],
            ax=ax2, palette=[colors['no-stim'], colors['stim']], hue_order=[0, 1], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Striatum') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Striatum') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Striatum') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Striatum') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax2.text(0, 11, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax2.text(1, 11, '*', ha='center', va='center', fontsize=12)
ax2.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Striatum', ylabel='State duration (s)',
        ylim=[0, 12])
ax2.legend().set_visible(False)

sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'Thalamus'],
            ax=ax3, palette=[colors['no-stim'], colors['stim']], hue_order=[0, 1], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Thalamus') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Thalamus') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Thalamus') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Thalamus') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax3.text(0, 3.5, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax3.text(1, 3.5, '*', ha='center', va='center', fontsize=12)
ax3.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Thalamus', ylabel='State duration (s)',
        ylim=[0, 6])
ax3.legend().set_visible(False)

"""
sns.boxplot(x='state', y='state_dur', hue='opto', data=state_feat_df[state_feat_df['region'] == 'Midbrain'],
            ax=ax4, palette=[colors['stim'], colors['no-stim']], hue_order=[1, 0], fliersize=0)
_, pd = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Midbrain') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 0), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Midbrain') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 0), 'state_dur'])
_, pu = mannwhitneyu(state_feat_df.loc[(state_feat_df['region'] == 'Midbrain') & (state_feat_df['opto'] == 0) & (state_feat_df['state'] == 1), 'state_dur'],
                     state_feat_df.loc[(state_feat_df['region'] == 'Midbrain') & (state_feat_df['opto'] == 1) & (state_feat_df['state'] == 1), 'state_dur'])
if pd < 0.05:
    ax4.text(0, 4, '*', ha='center', va='center', fontsize=12)
if pu < 0.05:
    ax4.text(1, 4, '*', ha='center', va='center')
ax4.set(xticks=[0, 1], xticklabels=['Down', 'Up'], xlabel='State', title='Midbrain', ylabel='State duration (s)',
        ylim=[0, 6])
ax4.legend().set_visible(False)
"""

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'Ephys', 'Anesthesia', 'UpDownState_comparison.jpg'), dpi=600)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

# Do ANOVA
no_opto_df = state_feat_df[(state_feat_df['opto'] == 0) & (state_feat_df['region'] != 'Hippocampus')
                           & (state_feat_df['region'] != 'Midbrain')]
mod = ols('state_dur ~ region', data=no_opto_df[no_opto_df['state'] == 1]).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(no_opto_df.loc[no_opto_df['state'] == 1, 'state_dur'],
                     no_opto_df.loc[no_opto_df['state'] == 1, 'region'])
tukey_upstate = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA up state p = {aov_table.loc["region", "PR(>F)"]}\n')
print(tukey_upstate)

mod = ols('state_dur ~ region', data=no_opto_df[no_opto_df['state'] == 0]).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
mc = MultiComparison(no_opto_df.loc[no_opto_df['state'] == 0, 'state_dur'],
                     no_opto_df.loc[no_opto_df['state'] == 0, 'region'])
tukey_downstate = mc.tukeyhsd(alpha=0.05)
print(f'\nANOVA down state p = {aov_table.loc["region", "PR(>F)"]}\n')
print(tukey_downstate)


sns.boxplot(x='region', y='state_dur', data=no_opto_df[no_opto_df['state'] == 1],
            ax=ax1, fliersize=0, color=colors['enhanced'])
ax1.set(ylim=[0, 10], title='Up states', xlabel='', ylabel='State duration (s)')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')

sns.boxplot(x='region', y='state_dur', data=no_opto_df[no_opto_df['state'] == 0],
            ax=ax2, fliersize=0, color=colors['suppressed'])
ax2.set(ylim=[0, 10], title='Down states', xlabel='', ylabel='')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

sns.despine(trim=True)
plt.tight_layout()

# %%
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)

region_df = state_feat_df.groupby(['pid', 'opto', 'state', 'region']).median(numeric_only=True).reset_index()

sns.lineplot(x='opto', y='state_dur', data=region_df[region_df['region'] == 'Cortex'],
             hue='state', estimator=None, units='pid', style='pid', hue_order=[1, 0],
             palette=[colors['enhanced'], colors['suppressed']], legend=None, dashes=False,
             markers=['o']*int(region_df[region_df['region'] == 'Cortex'].shape[0]/2), ax=ax1)
ax1.set(ylabel='State duration (s)', xticks=[0, 1], xticklabels=['Spontaneous', 'Opto stim'],
        title='Cortex', xlabel='')

sns.lineplot(x='opto', y='state_dur', data=region_df[region_df['region'] == 'Striatum'],
             hue='state', estimator=None, units='pid', style='pid', hue_order=[1, 0],
             palette=[colors['enhanced'], colors['suppressed']], legend=None, dashes=False,
             markers=['o']*int(region_df[region_df['region'] == 'Cortex'].shape[0]/2), ax=ax2)
ax2.set(ylabel='State duration (s)', xticks=[0, 1], xticklabels=['Spontaneous', 'Opto stim'],
        title='Striatum', xlabel='')

sns.lineplot(x='opto', y='state_dur', data=region_df[region_df['region'] == 'Thalamus'],
             hue='state', estimator=None, units='pid', style='pid', hue_order=[1, 0],
             palette=[colors['enhanced'], colors['suppressed']], legend=None, dashes=False,
             markers=['o']*int(region_df[region_df['region'] == 'Cortex'].shape[0]/2), ax=ax3)
ax3.set(ylabel='State duration (s)', xticks=[0, 1], xticklabels=['Spontaneous', 'Opto stim'],
        title='Thalamus', xlabel='')

sns.lineplot(x='opto', y='state_dur', data=region_df[region_df['region'] == 'Amygdala'],
             hue='state', estimator=None, units='pid', style='pid', hue_order=[1, 0],
             palette=[colors['enhanced'], colors['suppressed']], legend=None, dashes=False,
             markers=['o']*int(region_df[region_df['region'] == 'Cortex'].shape[0]/2), ax=ax4)
ax4.set(ylabel='State duration (s)', xticks=[0, 1], xticklabels=['Spontaneous', 'Opto stim'],
        title='Amygdala', xlabel='')

sns.despine(trim=True)
plt.tight_layout()



