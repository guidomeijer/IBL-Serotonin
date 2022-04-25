#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from serotonin_functions import paths, figure_style, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
METRICS = ['pupil_diameter', 'paw_l', 'nose_tip']
LABELS = ['Pupil diameter', 'Paw movement', 'Nose movement']
BINSIZE = 0.04
T_BEFORE = 1
T_AFTER = 3
time_ax = np.arange(-T_BEFORE, T_AFTER, BINSIZE)

# Figure path
fig_path, save_path = paths(dropbox=True)
fig_path= join(fig_path, 'PaperPassive')

# Get all saved design matrices
all_dm = glob(join(save_path, 'GLM', 'dm_*'))

# Load in subjects
subjects = load_subjects()

# Create one longform dataframe for plotting
all_mot_df = pd.DataFrame()
for i, dm_file in enumerate(all_dm):
    this_opto_df = pd.read_pickle(dm_file)
    this_mot_df = pd.DataFrame()
    
    # Baseline subtract pupil size
    
    
    for j, metric in enumerate(METRICS):
        this_mot_df[metric] = [item for row in this_opto_df[metric] for item in row]
    this_mot_df['time'] = np.tile(time_ax, this_opto_df.shape[0])
    this_mot_df['subject'] = dm_file[-27:-18]
    this_mot_df['date'] = dm_file[-17:-7]
    this_mot_df['sert-cre'] = subjects.loc[subjects['subject'] == dm_file[-27:-18], 'sert-cre'].values[0]
    all_mot_df = pd.concat((all_mot_df, this_mot_df), ignore_index=True)
    
# Change labels for plotting
all_mot_df.loc[all_mot_df['sert-cre'] == 1, 'sert-cre'] ='SERT'
all_mot_df.loc[all_mot_df['sert-cre'] == 0, 'sert-cre'] ='WT'
    
# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
lplt = sns.lineplot(x='time', y='pupil_diameter', hue='sert-cre', data=all_mot_df, ax=ax1,
                    ci=68, hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']])
ax1.legend(frameon=False)
ax1.set(xlabel='Time (s)', ylabel='Pupil size (%)')

sns.lineplot(x='time', y='paw_l', hue='sert-cre', data=all_mot_df, ax=ax2,
             ci=68, hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']], legend=False)
ax2.set(xlabel='Time (s)', ylabel='Paw movement (a.u.)')
    
    
    
    
    
    
    
    
  
   


