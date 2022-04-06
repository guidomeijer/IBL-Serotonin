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
fig_path, save_path = paths()
subjects = load_subjects()

# Load in GLM output
all_glm_df = pd.read_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))
all_glm_df['region'] = remap(all_glm_df['acronym'])

# Add sert-cre
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_glm_df.loc[all_glm_df['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]

# Add opto modulated neurons
opto_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
all_glm_df = pd.merge(all_glm_df, opto_neurons, on=['subject', 'date', 'neuron_id', 'pid', 'region'])
all_glm_df = all_glm_df.drop(['Unnamed: 0', 'mod_index_early', 'mod_index_late', 'p_value',
                              'latency_zeta', 'latency_peak', 'latency_peak_hw'], axis=1)

# Get average motion regressor
all_glm_df['all_motion'] = all_glm_df[MOTION_REG].mean(axis=1)

# Transform into long form
long_glm_df = pd.melt(all_glm_df, ['subject', 'date', 'neuron_id', 'acronym', 'sert-cre', 'region',
                                   'score', 'modulated', 'eid', 'probe', 'pid', 'all_motion'])

# %% Plot results
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)
sns.boxplot(x='sert-cre', y='opto_stim', data=all_glm_df[all_glm_df['modulated'] == 1],
               palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75, ax=ax1)
ax1.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel='Variance explaned \n stimulation regressor',
        ylim=[0, 0.5])

sns.boxplot(x='sert-cre', y='all_motion', data=all_glm_df[all_glm_df['modulated'] == 1],
               palette=[colors['sert'], colors['wt']], order=[1, 0], fliersize=2, linewidth=0.75, ax=ax2)
ax2.set(xticklabels=['SERT', 'WT'], xlabel='', ylabel='Variance explaned \n motion regressors',
        ylim=[0, 0.5])

plt.tight_layout()
sns.despine(trim=True, offset=4)
plt.savefig(join(fig_path, 'Ephys', 'GLM', 'var_explained_mot_stim.jpg'), dpi=300)

# %%
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(5, 3), dpi=dpi)
sns.boxplot(x='variable', y='value',
            data=long_glm_df[(long_glm_df['modulated'] == 1) & (long_glm_df['sert-cre'] == 1)],
               fliersize=2, linewidth=0.75, color=colors['general'], ax=ax1)
ax1.set(xlabel='', ylabel='Variance explaned',
        ylim=[0, 0.5])
ax1.tick_params('x', labelrotation=90)

all_glm_df['ratio_motion'] = 1
all_glm_df['ratio_opto'] = all_glm_df['opto_stim'] / all_glm_df['all_motion']

ax2.bar([0, 1], [all_glm_df.loc[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1), 'ratio_motion'].mean(),
                 all_glm_df.loc[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1), 'ratio_opto'].mean()],
        yerr=[all_glm_df.loc[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1), 'ratio_motion'].std(),
              all_glm_df.loc[(all_glm_df['sert-cre'] == 1) & (all_glm_df['modulated'] == 1), 'ratio_opto'].std()],
        width=0.75, color='grey')
ax2.set(xticks=[0, 1], xticklabels=['Motion', 'Stimulation'], ylabel='Normalized explained var.')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'Ephys', 'GLM', 'var_explained_all-reg.jpg'), dpi=300)



