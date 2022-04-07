#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join, isfile
from dlc_functions import get_dlc_XYs, smooth_interpolate_signal_sg, get_raw_and_smooth_pupil_dia
from serotonin_functions import (query_ephys_sessions, load_passive_opto_times, make_bins, paths,
                                 load_wheel_velocity)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
OVERWRITE = False
BINSIZE = 0.04
T_BEFORE = 1
T_AFTER = 2
STIM_DUR = 1
DLC_MARKERS = ['nose_tip', 'paw_l', 'paw_r', 'tongue_end_l', 'tongue_end_r']
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions()

for i in rec.index.values:
    # Get session data
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if ~OVERWRITE & isfile(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle')):
        print(f'Data for {subject} {date} found')
        continue
    print(f'Processing {subject}, {date}')

    # Load in data
    opto_train_start, _ = load_passive_opto_times(eid, one=one)  # opto pulses
    if len(opto_train_start) == 0:
        continue
    try:
        video_times, XYs = get_dlc_XYs(one, eid)  # DLC
    except:
        print('Failed to load DLC')
        continue
    cam_times = one.load_datasets(eid, datasets=['_ibl_bodyCamera.times.npy',
                                                 '_ibl_leftCamera.times.npy',
                                                 '_ibl_rightCamera.times.npy'])[0]
    times_body, times_left, times_right = cam_times[0], cam_times[1], cam_times[2]

    # Smooth motion energy
    print('Smoothing motion energy traces')
    try:
        mot_energy = one.load_datasets(eid, datasets=['bodyCamera.ROIMotionEnergy.npy',
                                                      'leftCamera.ROIMotionEnergy.npy',
                                                      'rightCamera.ROIMotionEnergy.npy'])[0]
        mot_body, mot_left, mot_right = mot_energy[0], mot_energy[1], mot_energy[2]
        mot_body = smooth_interpolate_signal_sg(mot_body)
        mot_left = smooth_interpolate_signal_sg(mot_left)
        mot_right = smooth_interpolate_signal_sg(mot_right)
    except:
        print('Failed to get motion energy')
        mot_body = np.zeros(times_body.shape)
        mot_left = np.zeros(times_left.shape)
        mot_right = np.zeros(times_right.shape)

    # Get stim start and stop times
    opto_df = pd.DataFrame()
    opto_df['trial_start'] = opto_train_start - T_BEFORE
    opto_df['trial_end'] = opto_train_start + STIM_DUR + T_AFTER
    opto_df['opto_start'] = opto_train_start
    opto_df['opto_end'] = opto_train_start + STIM_DUR

    # Load in wheel velocity
    opto_df['wheel_velocity'] = load_wheel_velocity(eid, opto_df['trial_start'].values,
                                                    opto_df['trial_end'].values, BINSIZE, one=one)

    # Get pupil diameter
    print('Loading in pupil size')
    _, pupil_diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)
    pupil_diameter[np.isnan(pupil_diameter)] = 0
    opto_df['pupil_diameter'] = make_bins(pupil_diameter, times_left, opto_df['trial_start'],
                                          opto_df['trial_end'], BINSIZE)

    # Create displacement of DLC markers
    print('Smoothing DLC traces')
    diff_video_times = (video_times[1:] + video_times[:-1]) / 2
    for i, dlc_key in enumerate(DLC_MARKERS):
        this_dist = np.linalg.norm(XYs[dlc_key][1:] - XYs[dlc_key][:-1], axis=1)
        this_dlc = smooth_interpolate_signal_sg(this_dist)
        opto_df[dlc_key] = make_bins(this_dlc, diff_video_times, opto_df['trial_start'],
                                     opto_df['trial_end'], BINSIZE)

    # Add motion energy
    opto_df['motion_energy_body'] = make_bins(mot_body, times_body, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
    opto_df['motion_energy_left'] = make_bins(mot_left, times_left, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
    opto_df['motion_energy_right'] = make_bins(mot_right, times_right, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)

    # Save design matrix
    opto_df.to_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))
    print('Saved output')


