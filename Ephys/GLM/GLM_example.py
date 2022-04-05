#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import numpy as np
import brainbox.modeling.utils as mut
import brainbox.io.one as bbone
from brainbox.io.one import SpikeSortingLoader
import neurencoding.design_matrix as dm
from brainbox.modeling.linear import LinearGLM
from brainbox.modeling.poisson import PoissonGLM
from serotonin_functions import query_ephys_sessions
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

BINSIZE = 0.02
KERNLEN = 0.6
SHORT_KL = 0.4
NBASES = 10

rec = query_ephys_sessions()

eid = rec['eid'][1]
pid = rec['pid'][1]

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
asd
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