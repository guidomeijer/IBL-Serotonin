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
METRICS = ['pupil_diameter', 'paw_l', 'nose_tip', 'motion_energy_left']
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
    for j, metric in enumerate(METRICS):
        this_mot_df[metric] = [item for row in this_opto_df[metric] for item in row]
    this_mot_df['time'] = np.tile(time_ax, this_opto_df.shape[0])
    this_mot_df['trial'] = np.repeat(np.arange(this_opto_df.shape[0]) + 1, time_ax.shape[0])
    this_mot_df['subject'] = dm_file[-27:-18]
    this_mot_df['date'] = dm_file[-17:-7]
    this_mot_df['sert-cre'] = subjects.loc[subjects['subject'] == dm_file[-27:-18], 'sert-cre'].values[0]
    
    # Baseline subtract
    for t in np.unique(this_mot_df['trial']):
        this_mot_df.loc[this_mot_df['trial'] == t, 'pupil_baseline'] = (
            this_mot_df.loc[this_mot_df['trial'] == t, 'pupil_diameter'].values
            - this_mot_df.loc[(this_mot_df['trial'] == t) & (this_mot_df['time'] < 0), 'pupil_diameter'].mean())
        this_mot_df.loc[this_mot_df['trial'] == t, 'paw_l_baseline'] = (
            this_mot_df.loc[this_mot_df['trial'] == t, 'paw_l'].values
            - this_mot_df.loc[(this_mot_df['trial'] == t) & (this_mot_df['time'] < 0), 'paw_l'].mean())
    
    all_mot_df = pd.concat((all_mot_df, this_mot_df), ignore_index=True)
    
# Restructure df for plotting
plot_df = all_mot_df.groupby(['subject', 'time']).mean()
plot_df.loc[plot_df['sert-cre'] == 1, 'sert-cre'] ='SERT'
plot_df.loc[plot_df['sert-cre'] == 0, 'sert-cre'] ='WT'
    
# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=dpi)

lplt = sns.lineplot(x='time', y='pupil_baseline', hue='sert-cre', data=plot_df, ax=ax1, estimator=None,
                    units='subject',
                    hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']])
ax1.legend(frameon=False, loc='upper left')
ax1.set(xlabel='Time (s)', ylabel='Pupil size (%)', xticks=[-1, 0, 1, 2, 3], yticks=[-5, 0, 5, 10, 15])

sns.lineplot(x='time', y='paw_l_baseline', hue='sert-cre', data=plot_df, ax=ax2, estimator=None,
             units='subject', hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']],
             legend=False)
ax2.set(xlabel='Time (s)', ylabel='Paw movement (%)', xticks=[-1, 0, 1, 2, 3])

sns.lineplot(x='time', y='nose_tip', hue='sert-cre', data=plot_df, ax=ax3, estimator=None,
             units='subject', hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']],
             legend=False)
ax3.set(xlabel='Time (s)', ylabel='Nose movement (%)', xticks=[-1, 0, 1, 2, 3])

sns.lineplot(x='time', y='motion_energy_left', hue='sert-cre', data=plot_df, ax=ax4, estimator=None,
             units='subject', hue_order=['SERT', 'WT'], palette=[colors['sert'], colors['wt']],
             legend=False)
ax4.set(xlabel='Time (s)', ylabel='Video motion energy (%)', xticks=[-1, 0, 1, 2, 3])

plt.tight_layout()
sns.despine(trim=True)
    
    
    
    
    
    
    
    
  
   


