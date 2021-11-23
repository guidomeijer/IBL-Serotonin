#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_raw_and_smooth_pupil_dia
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from serotonin_functions import (load_trials, paths, query_opto_sessions, figure_style, load_subjects,
                                 behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
BEHAVIOR_CRIT = True
TIME_WIN = [-0.5, 0]
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

subjects = load_subjects(behavior=True)

# TESTING
#subjects = subjects[subjects['subject'] != 'ZFM-02181'].reset_index(drop=True)

all_trials = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids = query_opto_sessions(nickname, one=one)

    #eids = eids[-5:]

    if BEHAVIOR_CRIT:
        eids = behavioral_criterion(eids)

    # Loop over sessions
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids):
        print(f'Processing session {j+1} of {len(eids)}')

        # Load in trials and video data
        try:
            trials = load_trials(eid, laser_stimulation=True, one=one)
        except:
            print('could not load trials')
            continue
        if trials is None:
            continue
        if 'laser_stimulation' not in trials.columns.values:
            continue

        details = one.get_details(eid)
        date = details['date']

        # Get stimulation block transitions
        no_probe_trials = trials.copy()
        no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_stimulation'] = (
            no_probe_trials.loc[no_probe_trials['probe_trial'] == 1, 'laser_probability'].round())
        stim_block_trans = np.append([0], np.array(np.where(np.diff(no_probe_trials['laser_stimulation']) != 0)) + 1)

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(one, eid)
        except:
            print('Could not load video and/or DLC data')
            continue

        # If the difference between timestamps and video frames is too large, skip
        if np.abs(video_times.shape[0] - XYs['pupil_left_r'].shape[0]) > 10000:
            print('Timestamp mismatch, skipping..')
            continue

        # Get pupil diameter
        raw_diameter, diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)

        # Assume frames were dropped at the end
        if video_times.shape[0] > diameter.shape[0]:
            video_times = video_times[:diameter.shape[0]]
        elif diameter.shape[0] > video_times.shape[0]:
            diameter = diameter[:video_times.shape[0]]

        # Calculate percentage change
        diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                         / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100

        # Get trial triggered baseline subtracted pupil diameter
        for t, trial_start in enumerate(trials['stimOn_times']):
            this_diameter = np.nanmedian(diameter_perc[(video_times > (trial_start + TIME_WIN[0]))
                                                  & (video_times < (trial_start + TIME_WIN[1]))])
            trials.loc[t, 'pupil_diameter'] = this_diameter
            
        # Add to dataframe of all trials
        trials['subject'] = nickname
        trials['session'] = date
        all_trials = all_trials.append(trials) 


# %% Plot 

# Remove nans
all_trials = all_trials[~all_trials['reaction_times'].isnull()]
all_trials = all_trials[~all_trials['pupil_diameter'].isnull()]

# Remove outliers
all_trials = all_trials[all_trials['pupil_diameter'] < 150]

colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)

r, p = pearsonr(all_trials.loc[all_trials['subject'] == 'ZFM-02600', 'reaction_times'],
                all_trials.loc[all_trials['subject'] == 'ZFM-02600', 'pupil_diameter'])
sns.scatterplot(x='reaction_times', y='pupil_diameter', data=all_trials[all_trials['subject'] == 'ZFM-02600'], ax=ax1,
                palette=[colors['wt'], colors['sert']])
ax1.set(xlabel='Reaction time (s)', ylabel='Pupil size (%)', xscale='log')

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'TrailCorrelations.png'), dpi=300)

