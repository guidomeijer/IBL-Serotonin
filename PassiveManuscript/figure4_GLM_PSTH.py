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
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
svd = TruncatedSVD(n_components=2, random_state=42)
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Settings
MOTION_REG = ['wheel_velocity', 'nose', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r',
              'motion_energy_body', 'motion_energy_left', 'motion_energy_right', 'pupil_diameter']
OPTO_REG = ['opto_4_bases', 'opto_6_bases', 'opto_8_bases', 'opto_10_bases', 'opto_12_bases', 
            'opto_boxcar']
MIN_NEURONS = 5
EX_NEURON_1 = 145
EX_PID_1 = '17b1fed7-c04c-45f3-a474-c19a3d1c3ea8'
EX_NEURON_2 = 263
EX_PID_2 = 'ad7e0c6e-d8de-4771-986b-382d0ae9af3a'
EX_NEURON_3 = 177
EX_PID_3 = 'ad7e0c6e-d8de-4771-986b-382d0ae9af3a'

#EX_NEURON_3 = 211
#EX_PID_3 = '7a82c06b-0e33-454b-a98f-786a4024c1d0'

EX_NEURON_5 = 507
EX_PID_5 = 'ea3cefba-dda1-4f8e-bd85-97547181a660'
EX_NEURON_4 = 282
EX_PID_4 = '5712250f-7ea2-4f55-a7ac-15e855533225'
EX_NEURON_6 = 181
EX_PID_6 = '17b1fed7-c04c-45f3-a474-c19a3d1c3ea8'

"""
# two peaks (first one small)
EX_NEURON_1 = 217
EX_PID_1 = '32be7608-6d32-4c8e-af5e-beb298ba8c73'

# late peak
EX_NEURON_1 = 252
EX_PID_1 = 'eead949d-45e5-46ae-9b1d-646b280b4ecf'

# two peaks
EX_NEURON_1 = 900
EX_PID_1 = 'ad7e0c6e-d8de-4771-986b-382d0ae9af3a'
"""

# Initialize some things
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')
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

# Get maximum motion and opto regressor
all_glm_df['all_motion'] = all_glm_df[MOTION_REG].max(axis=1)
all_glm_df['opto_stim'] = all_glm_df[OPTO_REG].max(axis=1)

# Get ratio
all_glm_df['ratio_opto'] = ((all_glm_df['opto_stim'] - all_glm_df['all_motion'])
                            / (all_glm_df['opto_stim'] + all_glm_df['all_motion']))

# Load in PSTH data
psth_df = pd.read_pickle(join(save_path, 'psth.pickle'))

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
f = plt.figure(figsize=(4, 2), dpi=dpi)
axs = f.subplot_mosaic('EAB\nFAC\nGAD\n', gridspec_kw={'width_ratios':[1, 2, 1]})
sc = axs['A'].scatter(merged_df['dim_1'], merged_df['dim_2'], c=merged_df['ratio_opto'],
                      cmap='coolwarm', vmin=-1, vmax=1)
#ax1.legend(frameon=False, bbox_to_anchor=(1, 0.7))
axs['A'].axis('off')
cbar = plt.colorbar(sc, orientation="horizontal", pad=0.05, ax=axs['A'])
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_label('Ratio stimulation / motion')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_1) & (merged_df['pid'] == EX_PID_1)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], 36], [merged_df.loc[ex_neuron, 'dim_2'], 25],
              color='k')
axs['B'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['B'].plot([0, 0], axs['B'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['B'].axis('off')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_2) & (merged_df['pid'] == EX_PID_2)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], 36], [merged_df.loc[ex_neuron, 'dim_2'], -7],
              color='k')
axs['C'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['C'].plot([0, 0], axs['C'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['C'].axis('off')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_3) & (merged_df['pid'] == EX_PID_3)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], 36], [merged_df.loc[ex_neuron, 'dim_2'], -20],
              color='k')
axs['D'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['D'].plot([0, 0], axs['D'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['D'].plot([0, 1], [-0.5, -0.5], color='k', lw=0.5)
axs['D'].text(0.5, -5, '1s', ha='center')
axs['D'].axis('off')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_4) & (merged_df['pid'] == EX_PID_4)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], -38], [merged_df.loc[ex_neuron, 'dim_2'], 25],
              color='k')
axs['E'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['E'].plot([0, 0], axs['E'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['E'].axis('off')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_5) & (merged_df['pid'] == EX_PID_5)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], -38], [merged_df.loc[ex_neuron, 'dim_2'], 0],
              color='k')
axs['F'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['F'].plot([0, 0], axs['F'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['F'].axis('off')

ex_neuron = merged_df[(merged_df['neuron_id'] == EX_NEURON_6) & (merged_df['pid'] == EX_PID_6)].index.values[0]
axs['A'].plot([merged_df.loc[ex_neuron, 'dim_1'], -38], [merged_df.loc[ex_neuron, 'dim_2'], -20],
              color='k')
axs['G'].plot(merged_df.loc[ex_neuron, 'time'], merged_df.loc[ex_neuron, 'peth'])
axs['G'].plot([0, 0], axs['G'].get_ylim(), color='grey', ls='--', lw=0.75)
axs['G'].plot([0, 1], [-2, -2], color='k', lw=0.5)
axs['G'].text(0.5, -7, '1s', ha='center')
axs['G'].axis('off')

plt.tight_layout()
sns.despine(trim=True)

#plt.savefig(join(fig_path, 'GLM_PSTH.pdf'))


