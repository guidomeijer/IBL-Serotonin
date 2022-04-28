#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import neurencoding.utils as mut
import neurencoding.design_matrix as dm
from serotonin_functions import query_ephys_sessions, paths, figure_style

# Settings
SUBJECT = 'ZFM-03330'
DATE = '2022-02-18' 
PLOT = ['pupil_diameter', 'motion_energy_left', 'paw_l', 'nose_tip']
LABELS = ['Pupil diameter', 'Video motion energy', 'Paw movement', 'Nose movement']
BINSIZE = 0.04
T_BEFORE = 1
T_AFTER = 3
MOT_KERNLEN = 0.4
MOT_NBASES = 10
OPTO_KERNLEN = 1
OPTO_NBASES = 6
fig_path, save_path = paths(dropbox=True)
fig_path= join(fig_path, 'PaperPassive', 'figure4')

# Load in data
opto_df = pd.read_pickle(join(save_path, 'GLM', f'dm_{SUBJECT}_{DATE}.pickle'))

# Define what kind of data type each column is
vartypes = {
    'trial_start': 'timing',
    'trial_end': 'timing',
    'opto_start': 'timing',
    'opto_end': 'timing',
    'wheel_velocity': 'continuous',
    'nose_tip': 'continuous',
    'paw_l': 'continuous',
    'paw_r': 'continuous',
    'tongue_end_l': 'continuous',
    'tongue_end_r': 'continuous',
    'motion_energy_body': 'continuous',
    'motion_energy_left': 'continuous',
    'motion_energy_right': 'continuous',
    'pupil_diameter': 'continuous'
}

# Initialize design matrix
design = dm.DesignMatrix(opto_df, vartypes=vartypes, binwidth=BINSIZE)

# Get basis functions
motion_bases_func = mut.full_rcos(MOT_KERNLEN, MOT_NBASES, design.binf)
opto_bases_func = mut.full_rcos(OPTO_KERNLEN, OPTO_NBASES, design.binf)


# %% Plot
time_ax = np.linspace(-T_BEFORE, T_AFTER, opto_df['pupil_diameter'][0].shape[0])
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
for i, metric in enumerate(PLOT):
    norm_metric = opto_df[metric][1] / np.max(opto_df[metric][1])
    plt.plot(time_ax, norm_metric, label=LABELS[i])
ax1.add_patch(Rectangle((0, 0), 1, ax1.get_ylim()[1], color='royalblue', alpha=0.25, lw=0))
ax1.add_patch(Rectangle((0, 0), 1, -1, color='royalblue', alpha=0.25, lw=0))
plt.legend(frameon=False, bbox_to_anchor=(0.4, -0.2, 0.5, 0.2))
ax1.axis('off')
plt.tight_layout()
plt.savefig(join(fig_path, 'example_traces.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(0.25, 0.5), dpi=dpi)
line_styles = ['-', '--', '-.', ':', '-', '--']
for i in range(opto_bases_func.shape[1]):
    ax1.plot(opto_bases_func[:, i], ls=line_styles[i], color='grey')
ax1.axis('off')
plt.savefig(join(fig_path, 'example_basis_functions.pdf'))
  
   


