#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join, isfile
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

BINSIZE = 0.04
KERNLEN = 0.4
NBASES = 10
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions()

for i in rec.index.values:
    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if isfile(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle')):
        opto_df = pd.read_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))
        opto_df.index = range(1, opto_df.shape[0]+1)
        print(f'Loaded in design matrix for {subject} {date}')
    else:
        print(f'Could not find design matrix for {subject} {date}')
        continue

    # Define what kind of data type each column is
    vartypes = {
        'trial_start': 'timing',
        'trial_end': 'timing',
        'opto_start': 'timing',
        'opto_end': 'timing',
        'wheel_velocity': 'continuous',
        'pupil_diameter': 'continuous',
        'nose_tip': 'continuous',
        'paw_l': 'continuous',
        'paw_r': 'continuous',
        'tongue_end_l': 'continuous',
        'tongue_end_r': 'continuous',
        'motion_energy_body': 'continuous',
        'motion_energy_left': 'continuous',
        'motion_energy_right': 'continuous'
    }

    # Initialize design matrix
    design = dm.DesignMatrix(opto_df, vartypes=vartypes, binwidth=BINSIZE)

    # Build basis functions
    bases_func = mut.full_rcos(KERNLEN, NBASES, design.binf)

    # Add regressors
    design.add_covariate_boxcar('opto_stim', 'opto_start', 'opto_end',
                                desc='Optogenetic stimulation')
    design.add_covariate('wheel_velocity', opto_df['wheel_velocity'], bases_func, offset=-KERNLEN,
                         desc='Wheel velocity')
    design.add_covariate('pupil_diameter', opto_df['pupil_diameter'], bases_func, offset=-KERNLEN,
                         desc='Pupil diameter')
    design.add_covariate('nose', opto_df['nose_tip'], bases_func, offset=-KERNLEN,
                         desc='Nose tip')
    design.add_covariate('paw_l', opto_df['paw_l'], bases_func, offset=-KERNLEN,
                         desc='Left paw')
    design.add_covariate('paw_r', opto_df['paw_r'], bases_func, offset=-KERNLEN,
                         desc='Right paw')
    design.add_covariate('tongue_end_l', opto_df['tongue_end_l'], bases_func, offset=-KERNLEN,
                         desc='Left tongue')
    design.add_covariate('tongue_end_r', opto_df['tongue_end_r'], bases_func, offset=-KERNLEN,
                         desc='Right tongue')
    design.add_covariate('motion_energy_body', opto_df['motion_energy_body'], bases_func, offset=-KERNLEN,
                         desc='Motion energy body camera')
    design.add_covariate('motion_energy_left', opto_df['motion_energy_left'], bases_func, offset=-KERNLEN,
                         desc='Motion energy left camera')
    design.add_covariate('motion_energy_right', opto_df['motion_energy_right'], bases_func, offset=-KERNLEN,
                         desc='Motion energy right camera')
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