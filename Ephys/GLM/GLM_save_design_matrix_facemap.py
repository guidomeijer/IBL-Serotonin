#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import zscore
from os.path import join, isfile
from dlc_functions import get_dlc_XYs, smooth_interpolate_signal_sg, get_raw_and_smooth_pupil_dia
from serotonin_functions import (query_ephys_sessions, load_passive_opto_times, make_bins, paths,
                                 load_wheel_velocity)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
DATA_PATH = '/media/guido/Data2/Facemap'
#DATA_PATH = '/media/guido/IBLvideo2/5HT/Data/Raw'
N_DIM = 10  # Number of motion SVD dimensions to include
OVERWRITE = False
ZSCORE = True
BINSIZE = 0.04
T_BEFORE = 1
T_AFTER = 2
STIM_DUR = 1
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions(one=one)

# Get facemap processed sessions
ses = glob(join(DATA_PATH, '*proc.npy'))

for i, file_path in enumerate(ses):

    # Get session data
    subject = file_path[-34:-25]
    date = file_path[-24:-14]
    eid = rec.loc[(rec['subject'] == subject) & (rec['date'] == date), 'eid']
    if len(eid) > 0:
        eid = eid.values[0]
    else:
        continue

    # Load in facemap data
    try:
        facemap_dict = np.load(file_path, allow_pickle=True).item()
    except Exception:
        continue

    # Load in DLC
    try:
        cam_times, XYs = get_dlc_XYs(one, eid)  # DLC
    except:
        print('Failed to load DLC')
        continue

    if facemap_dict['motSVD'][1].shape[0] != cam_times.shape[0]:
        print(f'Frame mismatch! Skipping session {subject} {date}')
        continue

    if ~OVERWRITE & isfile(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle')):
        print(f'Data for {subject} {date} found')
        continue
    print(f'Processing {subject}, {date}')

    # Load in opto pulse times
    opto_train_start, _ = load_passive_opto_times(eid, one=one)  # opto pulses

    # Get stim start and stop times
    opto_df = pd.DataFrame()
    opto_df['trial_start'] = opto_train_start - T_BEFORE
    opto_df['trial_end'] = opto_train_start + STIM_DUR + T_AFTER
    opto_df['opto_start'] = opto_train_start
    opto_df['opto_end'] = opto_train_start + STIM_DUR

    # Get paw motion energy
    paw_dist = np.abs(np.linalg.norm(XYs['paw_l'][1:] - XYs['paw_l'][:-1], axis=1))
    paw_dist_smooth = smooth_interpolate_signal_sg(paw_dist, window=11, order=1)  # smooth
    paw_dist_smooth[np.isnan(paw_dist_smooth)] = 0 # Set NaN to 0
    paw_dist_smooth = np.concatenate((paw_dist_smooth, [paw_dist_smooth[-1]]))  # duplicate last frame
    paw_dist_zscore = zscore(paw_dist_smooth)

    # Get pupil diameter
    print('Loading in pupil size')
    _, pupil_diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)
    # Transform into %
    diameter_perc = ((pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
                     / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)) * 100
    diameter_perc[np.isnan(diameter_perc)] = 0 # Set NaN to 0
    diameter_zscore = zscore(diameter_perc)

    # Add to dataframe
    if ZSCORE:
        opto_df['paw_motion'] = make_bins(paw_dist_zscore, cam_times,
                                          opto_df['trial_start'], opto_df['trial_end'], BINSIZE)
        opto_df['pupil_diameter'] = make_bins(diameter_zscore, cam_times, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
        for j in range(N_DIM):
            opto_df[f'motSVD_dim{j}'] = make_bins(zscore(facemap_dict['motSVD'][1][:, j]), cam_times,
                                                  opto_df['trial_start'], opto_df['trial_end'],
                                                  BINSIZE)
    else:
        opto_df['paw_motion'] = make_bins(paw_dist_smooth, cam_times,
                                          opto_df['trial_start'], opto_df['trial_end'], BINSIZE)
        opto_df['pupil_diameter'] = make_bins(diameter_perc, cam_times, opto_df['trial_start'],
                                              opto_df['trial_end'], BINSIZE)
        for j in range(N_DIM):
            opto_df[f'motSVD_dim{j}'] = make_bins(facemap_dict['motSVD'][1][:, j], cam_times,
                                                  opto_df['trial_start'], opto_df['trial_end'],
                                                  BINSIZE)

    # Save design matrix
    opto_df.to_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))
    print('Saved output')

