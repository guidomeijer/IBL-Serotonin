#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import brainbox.modeling.utils as mut
from brainbox.core import TimeSeries
from brainbox.processing import sync
import brainbox.io.one as bbone
from brainbox.io.one import SpikeSortingLoader
import neurencoding.design_matrix as dm
from dlc_functions import get_dlc_XYs, smooth_interpolate_signal_sg
from brainbox.modeling.linear import LinearGLM
from brainbox.modeling.poisson import PoissonGLM
from serotonin_functions import query_ephys_sessions, load_passive_opto_times, make_bins
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
BINSIZE = 0.02
KERNLEN = 0.6
SHORT_KL = 0.4
NBASES = 10
T_BEFORE = 1
T_AFTER = 2
STIM_DUR = 1

# Query sessions
rec = query_ephys_sessions()

# For now only use this one
eid = rec['eid'][1]
pid = rec['pid'][1]

# Load in data
opto_train_start, _ = load_passive_opto_times(eid, one=one)  # opto pulses
video_times, XYs = get_dlc_XYs(one, eid)  # DLC
wheel = one.load_object(eid, 'wheel', collection='alf')  # wheel

# Create boxcar predictor for opto stim
time = np.linspace(0, T_BEFORE+T_AFTER+STIM_DUR, int((1/BINSIZE)*(T_BEFORE+T_AFTER+STIM_DUR)))
opto_bc = np.zeros(time.shape)
opto_bc[(time > T_BEFORE) & (time < T_BEFORE+STIM_DUR)] = 1

# Get stim start and stop times
opto_df = pd.DataFrame()
opto_df['trial_start'] = opto_train_start - T_BEFORE
opto_df['trial_end'] = opto_train_start + STIM_DUR + T_AFTER

# Create displacement of DLC markers
print('Smoothing DLC traces')
diff_video_times = (video_times[:1] + video_times[:-1]) / 2
for i, dlc_key in enumerate(XYs.keys()):
    this_dist = np.linalg.norm(XYs[dlc_key][1:] - XYs[dlc_key][:-1], axis=1)
    this_dlc = smooth_interpolate_signal_sg(this_dist)
    opto_df[dlc_key] = make_bins(this_dlc, diff_video_times, opto_df['trial_start'],
                                 opto_df['trial_end'], BINSIZE)

# Construct dataframe
opto_df['opto_stim'] = [opto_bc] * opto_df.shape[0]
opto_df['wheel_velocity'] = make_bins(wheel.position, wheel.timestamps, opto_df['trial_start'],
                                      opto_df['trial_end'], BINSIZE)


asd







# %%

# Load in a dataframe of trial-by-trial data
trialsdf = bbone.load_trials_df(eid, one, maxlen=2., t_before=0.4, t_after=0.6, ret_wheel=True,
                                wheel_binsize=BINSIZE)

# Define what kind of data type each column is
vartypes = {
    'choice': 'value',
    'probabilityLeft': 'value',
    'feedbackType': 'value',
    'feedback_times': 'timing',
    'contrastLeft': 'value',
    'contrastRight': 'value',
    'goCue_times': 'timing',
    'stimOn_times': 'timing',
    'trial_start': 'timing', 'trial_end': 'timing',
    'wheel_velocity': 'continuous'
}

# The following is not a sensible model per se of the IBL task, but illustrates each regressor type

# Initialize design matrix
design = dm.DesignMatrix(trialsdf, vartypes=vartypes, binwidth=BINSIZE)

# Build some basis functions
longbases = mut.full_rcos(KERNLEN, NBASES, design.binf)
shortbases = mut.full_rcos(SHORT_KL, NBASES, design.binf)

design.add_covariate_timing('stimL', 'stimOn_times', longbases,
                            cond=lambda tr: np.isfinite(tr.contrastLeft),
                            desc='Stimulus onset left side kernel')
"""
design.add_covariate('wheel', trialsdf['wheel_velocity'], shortbases, offset=-SHORT_KL,
                     desc='Anti-causal regressor for wheel velocity')
design.add_covariate_raw('wheelraw', trialsdf['wheel_velocity'], desc='Wheel velocity, no bases')
"""
design.compile_design_matrix()

# Now let's load in some spikes and fit them
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# We will build a linear model and a poisson model:
lm = LinearGLM(design, spikes.times, spikes.clusters, binwidth=BINSIZE, mintrials=1)
pm = PoissonGLM(design, spikes.times, spikes.clusters, binwidth=BINSIZE, mintrials=1)

# Running the .fit() method is enough to start the fitting procedure:
lm.fit()
pm.fit()

# After which we can assess the score of each model on our data:
lm.score()
pm.score()