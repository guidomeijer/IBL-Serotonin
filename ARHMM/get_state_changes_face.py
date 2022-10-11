#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:03:49 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
from os.path import join
import ssm
from ssm.plots import gradient_cmap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from brainbox.task.closed_loop import roc_single_event
from matplotlib.ticker import FormatStrFormatter
from glob import glob
import pandas as pd
import math
from brainbox.plot import peri_event_time_histogram
from brainbox.io.one import SpikeSortingLoader
from brainbox.singlecell import calculate_peths
from serotonin_functions import (figure_style, load_passive_opto_times, get_neuron_qc, remap, paths,
                                 query_ephys_sessions, load_subjects)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

K = 2    # number of discrete states
D = 25   # dimension of the observations
T_BEFORE = 1
T_AFTER = 4
BIN_SIZE = 0.15
SMOOTHING = 0
PRE_TIME = [0.5, 0]  # for modulation index
POST_TIME = [0, 0.5]
FM_DIR = '/media/guido/Data2/Facemap/'  # dir with facemap data
OVERWRITE = False
PLOT = True

# Get path
fig_path, save_path = paths()

# Get all processed facemap files
fm_files = glob(join(FM_DIR, '*_proc.npy'))

# Query sessions
rec = query_ephys_sessions(one=one)

# Get subjects
subjects = load_subjects()

if OVERWRITE:
    state_change_df = pd.DataFrame()
else:
    state_change_df = pd.read_csv(join(save_path, 'state_change_face.csv'))

for i, path in enumerate(fm_files):

    # Get session data
    subject = path[-40:-31]
    date = path[-30:-20]
    try:
        sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
        eid = one.search(subject=subject, date_range=date)[0]
    except:
        continue

    if not OVERWRITE:
        if eid in state_change_df['eid'].values:
            continue

    print(f'Starting {subject}, {date}')

    # Load in timestamp data
    try:
        times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
    except:
        continue

    # Load in facemap data
    fm_dict = np.load(path, allow_pickle=True).item()

    # Facemap data is the last part of the video
    fm_times = times[times.shape[0] - fm_dict['motSVD'][1].shape[0]:]

    # Load opto times
    try:
        opto_times, _ = load_passive_opto_times(eid, one=one)
    except:
        continue

    # Select part of recording starting just before opto onset
    motSVD = fm_dict['motSVD'][1][fm_times > opto_times[0] - 10, :D]
    fm_times = fm_times[fm_times > opto_times[0] - 10]

    if fm_times.shape[0] == 0:
        continue

    if np.sum(fm_times[-1] > opto_times) != opto_times.shape[0]:
        print('Mismatch!')
        continue

    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, D, observations="ar")
    arhmm.fit(motSVD[:, :D])
    zhat = arhmm.most_likely_states(motSVD[:, :D])

    # Get state change times
    state_changes = fm_times[np.concatenate((np.zeros(1), np.diff(zhat))) != 0]

    # Get state change PETH
    peths, _ = calculate_peths(state_changes, np.ones(state_changes.shape[0]), [1],
                               opto_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)

    # Add to df
    state_change_df = pd.concat((state_change_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'sert-cre': sert_cre,
        'change_rate': peths['means'][0], 'time': peths['tscale']})))

    if PLOT:
        colors, dpi = figure_style()
        f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        ax.add_patch(Rectangle((0, 0), 1, 100, color='royalblue', alpha=0.25, lw=0))
        ax.add_patch(Rectangle((0, 0), 1, -100, color='royalblue', alpha=0.25, lw=0))
        peri_event_time_histogram(state_changes, np.ones(state_changes.shape[0]), opto_times, 1,
                                  t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                  smoothing=SMOOTHING, include_raster=True,
                                  error_bars='sem', pethline_kwargs={'color': 'black', 'lw': 1},
                                  errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                  raster_kwargs={'color': 'black', 'lw': 0.3},
                                  eventline_kwargs={'lw': 0}, ax=ax)
        ytick_max = math.ceil(ax.get_ylim()[1]/2.)*2  # ceil to even number
        ax.set(ylabel='State change rate (changes/s)', xlabel='Time (s)',
               yticks=[0, ytick_max/2,  ytick_max],
               ylim=[ax.get_ylim()[0], ytick_max])
        # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.tight_layout()
        plt.savefig(join(fig_path, 'Behavior', 'FaceStateChange', f'{subject}_{date}.jpg'), dpi=600)
        plt.close(f)

    # Save result
    state_change_df.to_csv(join(save_path, 'state_change_face.csv'))

# Save result
state_change_df.to_csv(join(save_path, 'state_change_face.csv'))
