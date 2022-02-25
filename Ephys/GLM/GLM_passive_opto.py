#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join
import brainbox.modeling.utils as mut
import brainbox.io.one as bbone
from brainbox.io.one import SpikeSortingLoader
import neurencoding.design_matrix as dm
from brainbox.modeling.linear import LinearGLM
from brainbox.modeling.poisson import PoissonGLM
from serotonin_functions import query_ephys_sessions, paths
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

BINSIZE = 0.02
KERNLEN = 0.6
SHORT_KL = 0.4
NBASES = 10
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions()

for i in rec.index.values:
    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    # Load in a dataframe of trial-by-trial data
    opto_df = pd.read_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))

    # Define what kind of data type each column is
    vartypes = {
        'opto_stim': 'boxcar',
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