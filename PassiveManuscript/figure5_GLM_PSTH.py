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
from serotonin_functions import paths, load_subjects, remap, figure_style, high_level_regions
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
svd = TruncatedSVD(n_components=2, random_state=42)
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Initialize some things
MOTION_REG = ['wheel_velocity', 'nose', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r',
              'motion_energy_body', 'motion_energy_left', 'motion_energy_right', 'pupil_diameter']
MIN_NEURONS = 5
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive')
subjects = load_subjects()

# Load in GLM output
all_glm_df = pd.read_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))
all_glm_df['region'] = remap(all_glm_df['acronym'])
all_glm_df['full_region'] = remap(all_glm_df['acronym'], combine=True, abbreviate=False)

# Add sert-cre
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_glm_df.loc[all_glm_df['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]

# Set 0 regressors to NaN
all_glm_df.loc[all_glm_df['motion_energy_left'] < 0.00001, 'motion_energy_left'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_right'] < 0.00001, 'motion_energy_right'] = np.nan
all_glm_df.loc[all_glm_df['motion_energy_body'] < 0.00001, 'motion_energy_body'] = np.nan

# Get average motion regressor
all_glm_df['all_motion'] = all_glm_df[MOTION_REG].mean(axis=1)

# Get ratio
all_glm_df['ratio_opto'] = ((all_glm_df['opto_stim'] - all_glm_df['all_motion'])
                            / (all_glm_df['opto_stim'] + all_glm_df['all_motion']))

# Load in PSTH data
psth_df = pd.read_pickle(join(save_path, 'psth.pickle'))
#psth_df['high_level_region'] = high_level_regions(psth_df['acronym'])
#psth_df = psth_df[psth_df['high_level_region'] != 'root']

# Do dimensionality reduction on PSTHs
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
for i in range(all_psth.shape[0]):
    all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])  # normalize
dim_red_psth = tsne.fit_transform(all_psth)
psth_df['dim_1'] = dim_red_psth[:, 0]
psth_df['dim_2'] = dim_red_psth[:, 1]

# Merge GLM and PSTH dataframes
merged_df = pd.merge(all_glm_df, psth_df, on=['subject', 'date', 'neuron_id', 'pid'])

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3.2, 2), dpi=dpi)
#sc = sns.scatterplot(x='dim_1', y='dim_2', data=merged_df, c=merged_df['ratio_opto'], ax=ax1)
sc = ax1.scatter(merged_df['dim_1'], merged_df['dim_2'], c=merged_df['ratio_opto'], cmap='coolwarm')
#ax1.legend(frameon=False, bbox_to_anchor=(1, 0.7))
plt.colorbar(sc)
ax1.set(xlabel='PC 1', ylabel='PC 2')
plt.tight_layout()
sns.despine(trim=True)


