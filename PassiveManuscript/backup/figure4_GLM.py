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
from scipy.stats import mannwhitneyu
from serotonin_functions import paths, load_subjects, remap, figure_style, combine_regions

# Initialize some things
""" 
MOTION_REG = ['wheel_velocity', 'nose', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r',
              'motion_energy_body', 'motion_energy_left', 'motion_energy_right', 'pupil_diameter']
OPTO_REG = ['opto_4_bases', 'opto_6_bases', 'opto_8_bases', 'opto_10_bases', 'opto_12_bases', 
            'opto_boxcar']
"""

MOTION_REG = ['nose', 'paw_l', 'tongue_end_l', 'pupil_diameter']
OPTO_REG = ['opto_onset', 'opto_boxcar']

MIN_NEURONS_RATIO = 5
MIN_NEURONS_PERC = 20
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')
subjects = load_subjects()

# Load in GLM output
all_glm_df = pd.read_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))

# Add sert-cre
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_glm_df.loc[all_glm_df['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]
    
# Add regions
all_glm_df['region'] = remap(all_glm_df['acronym'])

# Add opto modulated neurons
opto_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_glm_df = pd.merge(all_glm_df, opto_neurons, on=['subject', 'date', 'neuron_id', 'pid', 'region'])

# Add full region
all_glm_df['full_region'] = combine_regions(all_glm_df['region'])

# Drop ZFM-01802 for now
all_glm_df = all_glm_df[all_glm_df['subject'] != 'ZFM-01802']

# Drop root
all_glm_df = all_glm_df[all_glm_df['region'] != 'root']

"""
# Set 0 regressors to NaN
all_glm_df.loc[all_glm_df['motion_energy_left'] < 0.00001, 'motion_energy_left'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_right'] < 0.00001, 'motion_energy_right'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_body'] < 0.00001, 'motion_energy_body'] = np.nan
"""

# Get maximum motion and opto regressor
all_glm_df['all_motion'] = all_glm_df[MOTION_REG].max(axis=1)
all_glm_df['opto_stim'] = all_glm_df[OPTO_REG].max(axis=1)

# Get ratio
all_glm_df['ratio_opto'] = ((all_glm_df['opto_stim'] - all_glm_df['all_motion'])
                            / (all_glm_df['opto_stim'] + all_glm_df['all_motion']))

# Get enhanced and suppressed
all_glm_df['enhanced_late'] = all_glm_df['modulated'] & (all_glm_df['mod_index_late'] > 0)
all_glm_df['suppressed_late'] = all_glm_df['modulated'] & (all_glm_df['mod_index_late'] < 0)

# Transform into long form
long_glm_df = pd.melt(all_glm_df, ['subject', 'date', 'neuron_id', 'acronym', 'sert-cre', 'region', 'ratio_opto',
                                   'full_region', 'score', 'modulated', 'eid', 'probe', 'pid', 'all_motion'])

# Perform statistics
_, p_mod = mannwhitneyu(all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 1), 'score'],
                        all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 0), 'score'])
_, p_all = mannwhitneyu(all_glm_df.loc[all_glm_df['sert-cre'] == 1, 'score'],
                        all_glm_df.loc[all_glm_df['sert-cre'] == 0, 'score'])
print(f'p-value modulated neurons: {p_mod}')
print(f'p-value all neurons: {p_all}')

_, p_stim = mannwhitneyu(all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 1), 'opto_stim'],
                         all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 0), 'opto_stim'])
_, p_motion = mannwhitneyu(all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 1), 'all_motion'],
                           all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 0), 'all_motion'])
print(f'p-value opto stim: {p_stim}')
print(f'p-value motion: {p_motion}')

# Get dataframe with means per animal
grouped_df = all_glm_df[all_glm_df['modulated']].groupby('subject').mean()

# Calculate percentage stim modulated neurons that are better explained by stim vs motion
all_glm_df['better_stim'] = (all_glm_df['ratio_opto'] > 0) & (all_glm_df['modulated'])
summary_df = all_glm_df[(all_glm_df['sert-cre'] == 1)].groupby(['full_region']).sum()
summary_df['n_neurons'] = all_glm_df.groupby(['full_region']).size()
summary_df = summary_df.reset_index()
summary_df = summary_df[summary_df['full_region'] != 'root']
summary_df['perc_enh_late'] =  (summary_df['enhanced_late'] / summary_df['n_neurons']) * 100
summary_df['perc_supp_late'] =  (summary_df['suppressed_late'] / summary_df['n_neurons']) * 100
summary_df['perc_mod'] =  (summary_df['better_stim'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_NEURONS_PERC]
summary_df['perc_supp_late'] = -summary_df['perc_supp_late']
ordered_regions = summary_df.sort_values('perc_mod', ascending=False).reset_index()

# %% Plot overall variance explained

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
sns.boxplot(x='sert-cre', y='score', data=all_glm_df[all_glm_df['modulated'] == 1], ax=ax1,
            palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=0, linewidth=0.75)
ax1.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel='Variance explained by full model',
        ylim=[0, 0.2])

plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'var_ex_full_model.pdf'))

# %% Plot variance explained by motion and opto

colors, dpi = figure_style()
glm_df_long = all_glm_df[all_glm_df['modulated'] == 1].melt(['pid', 'neuron_id'], ['opto_stim', 'all_motion'])
f, ax1 = plt.subplots(1, 1, figsize=(1.2, 1.75), dpi=dpi)
sns.boxplot(x='variable', y='value', data=glm_df_long,
            palette=[colors['glm_motion'], colors['glm_stim']],
            order=['all_motion', 'opto_stim'], fliersize=0, linewidth=0.75, ax=ax1)
ax1.set(xticklabels=['Motion', 'Stim.'], xlabel='', ylabel=u'Δ variance explained',
        ylim=[0.00001, 0.1], yscale='log')

plt.tight_layout()
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'var_explained_mot_stim.pdf'))

"""
# %% Plot variance explained by motion and opto
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi)
sns.boxplot(x='sert-cre', y='opto_stim', data=all_glm_df[all_glm_df['modulated'] == 1],
            palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=0, linewidth=0.75, ax=ax1)
ax1.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel=u'Δ var. explained by stimulation',
        ylim=[0.0001, 0.1], yscale='log')

sns.boxplot(x='sert-cre', y='all_motion', data=all_glm_df[all_glm_df['modulated'] == 1],
               palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=0, linewidth=0.75, ax=ax2)
ax2.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel=u'Δ var. explained by motion',
        ylim=[0.0001, 0.1], yscale='log')

plt.tight_layout(w_pad=4)
sns.despine(trim=True, offset=3)
plt.savefig(join(fig_path, 'var_explained_mot_stim.pdf'))
"""

# %% Plot histogram of motion / stim ratio
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

ax1.hist(all_glm_df.loc[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 1), 'ratio_opto'],
         bins=15, color='grey')
ax1.set(ylabel='Neuron count', xlabel='Ratio stimulation / motion', ylim=[0, 120],
        xticks=[-1, -0.5, 0, 0.5, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'ratio_motion_stim.pdf'))

# %% Plot ratio per region

# Drop root and only keep modulated neurons
glm_df_slice = all_glm_df[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1)
                          & (all_glm_df['full_region'] != 'root')]
grouped_df = glm_df_slice.groupby('full_region').size()
grouped_df = grouped_df[grouped_df >= MIN_NEURONS_RATIO]
glm_df_slice = glm_df_slice[glm_df_slice['full_region'].isin(grouped_df.index.values)]

sort_regions = glm_df_slice.groupby('full_region').mean()['ratio_opto'].sort_values().index.values

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
ax1.plot([0, 0], [-1, glm_df_slice.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.boxplot(x='ratio_opto', y='full_region', color='orange', ax=ax1,
            data=glm_df_slice, order=sort_regions, fliersize=0, linewidth=0.75)
ax1.set(ylabel='', xlabel='Ratio stimulation / motion', xlim=[-1, 1],
        xticks=[-1, -0.5, 0, 0.5, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'ratio_opto_motion_per_region.pdf'))

# %% Plot variance explained by opto per region

# Drop root and only keep modulated neurons
glm_df_slice = all_glm_df[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1)
                          & (all_glm_df['full_region'] != 'root')]
grouped_df = glm_df_slice.groupby('full_region').size()
grouped_df = grouped_df[grouped_df >= MIN_NEURONS_RATIO]
glm_df_slice = glm_df_slice[glm_df_slice['full_region'].isin(grouped_df.index.values)]

sort_regions = glm_df_slice.groupby('full_region').median()['opto_stim'].sort_values(ascending=False).index.values

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
ax1.plot([0, 0], [-1, glm_df_slice.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.boxplot(x='opto_stim', y='full_region', color='orange', ax=ax1,
            data=glm_df_slice, order=sort_regions, fliersize=0, linewidth=0.75)
ax1.set(ylabel='', xlabel=u'Δ var. explained by stimulation', xscale='log', xlim=[0.001, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'opto_per_region.pdf'))

# %% Plot variance explained by motion per region

# Drop root and only keep modulated neurons
glm_df_slice = all_glm_df[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1)
                          & (all_glm_df['full_region'] != 'root')]
grouped_df = glm_df_slice.groupby('full_region').size()
grouped_df = grouped_df[grouped_df >= MIN_NEURONS_RATIO]
glm_df_slice = glm_df_slice[glm_df_slice['full_region'].isin(grouped_df.index.values)]

sort_regions = glm_df_slice.groupby('full_region').median()['all_motion'].sort_values(ascending=False).index.values

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
ax1.plot([0, 0], [-1, glm_df_slice.shape[0]], color=[0.5, 0.5, 0.5], ls='--')
sns.boxplot(x='all_motion', y='full_region', color='orange', ax=ax1,
            data=glm_df_slice, order=sort_regions, fliersize=0, linewidth=0.75)
ax1.set(ylabel='', xlabel=u'Δ var. explained by motion', xscale='log', xlim=[0.001, 1])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'motion_per_region.pdf'))

# %% Plot var explained by opto vs motion per neuron in scatterplot

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.scatter(all_glm_df['opto_stim'], all_glm_df['all_motion'], c=colors['general'], s=2)
ax1.plot([0.0001, 1], [0.0001, 1], color='k')
ax1.set(xlabel=u'Δ var. explained by stimulation', ylabel=u'Δ var. explained by motion',
        xlim=[0.0001, 1], ylim=[0.0001, 1], xscale='log', yscale='log',
        xticks=[0.0001, 0.01, 1], yticks=[0.0001, 0.01, 1])
plt.minorticks_off()
plt.tight_layout()
sns.despine(trim=True)

# %% Plot percentage modulated neurons but only the ones that are better explained by opto
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi)
sns.barplot(x='perc_mod', y='full_region', data=summary_df.sort_values('perc_mod', ascending=False),
            color=colors['general'], ax=ax1)
ax1.text(22, 7, 'Stim. / Mot. > 0', ha='center')
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 40], xticks=np.arange(0, 41, 10))
#ax1.plot([-1, ax1.get_xlim()[1]], [5, 5], ls='--', color='grey')
#plt.xticks(rotation=90)
#ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_path, 'perc_modulated_neurons_per_region.pdf'))


"""
# %%
glm_df_slice = all_glm_df[(all_glm_df['modulated'] == 1) & (all_glm_df['sert-cre'] == 1)]
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.scatter(glm_df_slice['ratio_opto'], glm_df_slice['mod_index_early'])
ax1.scatter(glm_df_slice['ratio_opto'], glm_df_slice['mod_index_late'])
ax1.set(xlabel='Ratio stimulation / motion', ylabel='Modulation index', xlim=[-1, 1], ylim=[-1, 1])
plt.tight_layout()
sns.despine(trim=True, offset=2)

# %%
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)

sns.stripplot(x='perc_enh_late', y='full_region', data=summary_df, order=ordered_regions['full_region'],
              color='k', alpha=0, ax=ax1)  # this actually doesn't plot anything
ax1.plot([0, 0], [0, summary_df.shape[0]], color=[0.5, 0.5, 0.5])

ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=0, xmax=ordered_regions['perc_enh_late'],
           color=colors['enhanced'])
ax1.hlines(y=np.arange(ordered_regions.shape[0]), xmin=ordered_regions['perc_supp_late'], xmax=0,
           color=colors['suppressed'])
ax1.plot(ordered_regions['perc_supp_late'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['suppressed'])
ax1.plot(ordered_regions['perc_enh_late'], np.arange(ordered_regions.shape[0]), 'o',
         color=colors['enhanced'])
ax1.set(ylabel='', xlabel='Modulated neurons (%)', xlim=[-60, 40],
        xticklabels=np.concatenate((np.arange(60, 0, -20), np.arange(0, 41, 20))))
ax1.spines['bottom'].set_position(('data', summary_df.shape[0]))
ax1.margins(x=0)

plt.tight_layout()
sns.despine(trim=True)
"""

