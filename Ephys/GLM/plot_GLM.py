#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:25:11 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from serotonin_functions import paths, load_subjects, remap, figure_style

# Initialize some things
MOTION_REG = ['wheel_velocity', 'nose', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r',
              'motion_energy_body', 'motion_energy_left', 'motion_energy_right', 'pupil_diameter']
MIN_NEURONS = 5
fig_path, save_path = paths()
subjects = load_subjects()

# Load in GLM output
all_glm_df = pd.read_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))
all_glm_df['region'] = remap(all_glm_df['acronym'])
all_glm_df['full_region'] = remap(all_glm_df['acronym'], combine=True, abbreviate=False)

# Add sert-cre
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_glm_df.loc[all_glm_df['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]

# Drop root and void
all_glm_df = all_glm_df.reset_index(drop=True)
all_glm_df = all_glm_df.drop(index=[i for i, j in enumerate(all_glm_df['region']) if 'root' in j])
all_glm_df = all_glm_df.reset_index(drop=True)
all_glm_df = all_glm_df.drop(index=[i for i, j in enumerate(all_glm_df['region']) if 'void' in j])

# Add opto modulated neurons
opto_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_glm_df = pd.merge(all_glm_df, opto_neurons, on=['subject', 'date', 'neuron_id', 'pid', 'region'])
all_glm_df = all_glm_df.drop(['Unnamed: 0', 'mod_index_early', 'mod_index_late', 'p_value',
                              'latency_zeta', 'latency_peak', 'latency_peak_hw'], axis=1)

# Set 0 regressors to NaN
all_glm_df.loc[all_glm_df['motion_energy_left'] < 0.00001, 'motion_energy_left'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_right'] < 0.00001, 'motion_energy_right'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_body'] < 0.00001, 'motion_energy_body'] = np.nan

# Get average motion regressor
all_glm_df['all_motion'] = all_glm_df[MOTION_REG].mean(axis=1)

# Get ratio
all_glm_df['ratio_opto'] = ((all_glm_df['opto_stim'] - all_glm_df['all_motion'])
                            / (all_glm_df['opto_stim'] + all_glm_df['all_motion']))

# Transform into long form
long_glm_df = pd.melt(all_glm_df, ['subject', 'date', 'neuron_id', 'acronym', 'sert-cre', 'region', 'ratio_opto',
                                   'full_region', 'score', 'modulated', 'eid', 'probe', 'pid', 'all_motion'])

# %% Plot results
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 4), dpi=dpi)
sns.boxplot(x='sert-cre', y='opto_stim', data=all_glm_df[all_glm_df['modulated'] == 1],
            palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75, ax=ax1)
ax1.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel=u'Δ variance explaned by stimulation',
        ylim=[0.00001, 1], yscale='log')

sns.boxplot(x='sert-cre', y='all_motion', data=all_glm_df[all_glm_df['modulated'] == 1],
               palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75, ax=ax2)
ax2.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel=u'Δ variance explaned by motion',
        ylim=[0.00001, 1], yscale='log')

sns.boxplot(x='sert-cre', y='score', data=all_glm_df[all_glm_df['modulated'] == 1], ax=ax3,
            palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75)
ax3.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel='Variance explaned by full model', ylim=[0, 0.71],
        title='5-HT modulated neurons')

sns.boxplot(x='sert-cre', y='score', data=all_glm_df, ax=ax4,
            palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75)
ax4.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel='Variance explaned by full model', ylim=[0, 0.71],
        title='All neurons')

plt.tight_layout(pad=2)
sns.despine(trim=True, offset=4)
plt.savefig(join(fig_path, 'Ephys', 'GLM', 'var_explained_mot_stim.jpg'), dpi=300)

# %%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 2, 2]}, figsize=(8, 3), dpi=dpi)
sns.boxplot(x='variable', y='value',
            data=long_glm_df[(long_glm_df['modulated'] == 1) & (long_glm_df['sert-cre'] == 1)],
               fliersize=2, linewidth=0.75, color=colors['general'], ax=ax1)
ax1.set(xlabel='', ylabel=u'Δ variance explaned', ylim=[0.00001, 1], yscale='log')
ax1.tick_params('x', labelrotation=90)

ax2.hist(all_glm_df.loc[all_glm_df['modulated'] == 1, 'ratio_opto'], bins=25, color='grey')
ax2.set(ylabel='Neuron count', xlabel='Ratio stimulation / motion', title='5-HT modulated neurons',
        ylim=[0, 40])

ax3.hist(all_glm_df['ratio_opto'], bins=25, color='grey')
ax3.set(ylabel='Neuron count', xlabel='Ratio stimulation / motion', title='All neurons')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'GLM', 'var_explained_all-reg.jpg'), dpi=300)

# %%

# Drop root and only keep modulated neurons
glm_df_slice = all_glm_df[(all_glm_df['full_region'] != 'root') & (all_glm_df['modulated'] == 1)]
grouped_df = glm_df_slice.groupby('full_region').size()
grouped_df = grouped_df[grouped_df >= MIN_NEURONS]
glm_df_slice = glm_df_slice[glm_df_slice['full_region'].isin(grouped_df.index.values)]

sort_regions = glm_df_slice.groupby('full_region').mean()['ratio_opto'].sort_values().index.values

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
sns.barplot(x='ratio_opto', y='full_region', color='orange', ax=ax1,
            data=glm_df_slice, order=sort_regions, ci=68)
ax1.set(ylabel='', xlabel='Ratio stimulation / motion', xlim=[-1, 1])

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'Ephys', 'GLM', 'ratio_opto_motion_per_region.jpg'), dpi=300)



