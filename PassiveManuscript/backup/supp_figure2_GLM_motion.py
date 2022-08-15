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
FIG = 2
MIN_NEURONS = 10
MOTION_REG = ['wheel_velocity', 'nose', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r',
              'motion_energy_body', 'motion_energy_left', 'motion_energy_right', 'pupil_diameter']
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'supp_figure2')
subjects = load_subjects()

# Load in GLM output
all_glm_df = pd.read_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))

# Add sert-cre
for i, nickname in enumerate(np.unique(subjects['subject'])):
    all_glm_df.loc[all_glm_df['subject'] == nickname, 'sert-cre'] = subjects.loc[
        subjects['subject'] == nickname, 'sert-cre'].values[0]

# Add regions
all_glm_df['region'] = remap(all_glm_df['acronym'])
all_glm_df['full_region'] = combine_regions(all_glm_df['region'])

# Drop root
all_glm_df = all_glm_df[all_glm_df['region'] != 'root']

# Get max motion regressor
all_glm_df['max_motion'] = all_glm_df[MOTION_REG].max(axis=1)

# %%

# Drop root and only keep modulated neurons
glm_df_slice = all_glm_df[(all_glm_df['sert-cre'] == 1) & (all_glm_df['full_region'] != 'root')]
grouped_df = glm_df_slice.groupby('full_region').size()
grouped_df = grouped_df[grouped_df >= MIN_NEURONS]
glm_df_slice = glm_df_slice[glm_df_slice['full_region'].isin(grouped_df.index.values)]

sort_regions = glm_df_slice.groupby('full_region').mean()['max_motion'].sort_values(
    ascending=False).index.values

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi)
sns.boxplot(x='max_motion', y='full_region', color='orange', ax=ax1,
            data=glm_df_slice, order=sort_regions, fliersize=0)
ax1.set(ylabel='', xlabel=u'Δ var. explained by motion', xlim=[0, 0.3],
        xticks=[0, .1, .2, .3])

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(fig_path, 'GLM_motion_per_region.pdf'))
