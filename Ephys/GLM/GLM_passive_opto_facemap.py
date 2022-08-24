#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:10:21 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
from os.path import join, isfile
import neurencoding.utils as mut
from brainbox.io.one import SpikeSortingLoader
import neurencoding.design_matrix as dm
from neurencoding.linear import LinearGLM
from neurencoding.poisson import PoissonGLM
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import (query_ephys_sessions, paths, get_artifact_neurons, remap,
                                 get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
OVERWRITE = True
BINSIZE = 0.04
MOT_KERNLEN = 0.4
MOT_NBASES = 3
OPTO_KERNLEN = 1
OPTO_NBASES = 3
fig_path, save_path = paths()

# Query sessions
rec = query_ephys_sessions()

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

all_glm_df = pd.DataFrame()
for i in rec.index.values:
    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    if isfile(join(save_path, 'GLM', f'lm_scores_{subject}_{date}.csv')) & ~OVERWRITE:
        print(f'\nFound GLM results for {subject} {date}')
        continue
    if isfile(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle')):
        opto_df = pd.read_pickle(join(save_path, 'GLM', f'dm_{subject}_{date}.pickle'))
        print(f'\nLoaded in design matrix for {subject} {date}')
    else:
        print(f'\nCould not find design matrix for {subject} {date}')
        continue


    # Define what kind of data type each column is
    vartypes = {
        'trial_start': 'timing',
        'trial_end': 'timing',
        'opto_start': 'timing',
        'opto_end': 'timing',
        'paw_motion': 'continuous',
        'pupil_diameter': 'continuous',
        'motSVD_dim0': 'continuous',
        'motSVD_dim1': 'continuous',
        'motSVD_dim2': 'continuous',
        'motSVD_dim3': 'continuous',
        'motSVD_dim4': 'continuous',
        'motSVD_dim5': 'continuous',
        'motSVD_dim6': 'continuous',
        'motSVD_dim7': 'continuous',
        'motSVD_dim8': 'continuous',
        'motSVD_dim9': 'continuous',
    }

    # Initialize design matrix
    design = dm.DesignMatrix(opto_df, vartypes=vartypes, binwidth=BINSIZE)

    # Build basis functions
    motion_bases_func = mut.full_rcos(MOT_KERNLEN, MOT_NBASES, design.binf)
    opto_bases_funcs = mut.full_rcos(OPTO_KERNLEN, OPTO_NBASES, design.binf)

    # Add regressors
    design.add_covariate_timing('opto_onset', 'opto_start', opto_bases_funcs, desc='Optogenetic stimulation')
    design.add_covariate_timing('opto_offset', 'opto_end', opto_bases_funcs, desc='Optogenetic stimulation')
    design.add_covariate_boxcar('opto_boxcar', 'opto_start', 'opto_end', desc='Optogenetic stimulation')
    design.add_covariate('paw_motion', opto_df['paw_motion'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Paw motion')
    design.add_covariate('pupil_diameter', opto_df['pupil_diameter'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Pupil diameter')
    design.add_covariate('motSVD_dim0', opto_df['motSVD_dim0'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim1', opto_df['motSVD_dim1'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim2', opto_df['motSVD_dim2'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim3', opto_df['motSVD_dim3'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim4', opto_df['motSVD_dim4'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    """
    design.add_covariate('motSVD_dim5', opto_df['motSVD_dim5'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim6', opto_df['motSVD_dim6'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim7', opto_df['motSVD_dim7'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim8', opto_df['motSVD_dim8'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    design.add_covariate('motSVD_dim9', opto_df['motSVD_dim9'], motion_bases_func, offset=-MOT_KERNLEN,
                         desc='Motion SVD')
    """

    design.compile_design_matrix()
    print('Compiled design matrix')

    # Load in the neural data
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except:
        continue

    # Apply neuron QC and exclude artifact units
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    clusters['region'] = remap(clusters['atlas_id'], combine=True)

    # Build a linear model
    print('Fitting linear model')
    try:
        lm = LinearGLM(design, spikes.times, spikes.clusters, binwidth=BINSIZE, mintrials=1)
        lm.fit()
        lm_scores = lm.score()
        sfs = mut.SequentialSelector(lm)
        sfs.fit(progress=False)
        glm_results = sfs.deltas_
        glm_results['score'] = lm_scores
        glm_results['acronym'] = clusters['acronym'][glm_results.index]
        glm_results['subject'] = subject
        glm_results['date'] = date
        glm_results['pid'] = pid
        glm_results = glm_results.reset_index()
        glm_results = glm_results.rename({'index': 'neuron_id'}, axis=1)
        all_glm_df = pd.concat((all_glm_df, glm_results), ignore_index=True)
        all_glm_df.to_csv(join(save_path, 'GLM', 'GLM_passive_opto.csv'))
    except:
        print('\nFailed to fit GLM model\n')
        continue


