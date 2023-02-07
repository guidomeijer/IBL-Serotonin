import sys
import os
from os.path import join
import autograd.numpy as np
from glob import glob
from serotonin_functions import paths
from glm_hmm_utils import (load_cluster_arr, load_session_fold_lookup,
                           load_data, create_violation_mask, launch_glm_hmm_job)

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

# Settings
Ks = [2, 5]  # number of latent states
D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations
global_fit = True
# perform mle => set transition_alpha to 1
transition_alpha = 1
prior_sigma = 100
N_initializations = 20  # Number of times to initialize the fit

# Paths
_, data_path = paths()
data_dir = join(data_path, 'GLM-HMM')
results_dir = join(data_path, 'GLM-HMM', 'results')

#  read in data and train/test split
animal_file = join(data_dir, 'all_animals_concat.npz')
session_fold_lookup_table = load_session_fold_lookup(join(
    data_dir, 'all_animals_concat_session_fold_lookup.npz'))

inpt, y, session = load_data(animal_file)
#  append a column of ones to inpt to represent the bias covariate:
inpt = np.hstack((inpt, np.ones((len(inpt),1))))
y = y.astype('int')
# Identify violations for exclusion:
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])

# Get directories with data for each fold
fold_dirs = glob(join(results_dir, 'GLM', 'fold_*'))

# Loop over K, initializations and folds
for ki, K in enumerate(Ks):
    for kk, fold_dir in enumerate(fold_dirs):
        print(f'Starting fold {kk+1} of {len(fold_dirs)}')
        for i, iter in enumerate(range(N_initializations)):
            print(f'Initialization {i+1} of {N_initializations}')
            #  GLM weights to use to initialize GLM-HMM
            init_param_file = join(fold_dir, 'variables_of_interest_iter_0.npz')

            # create save directory for this initialization/fold combination:
            save_directory = join(results_dir, 'GLM_HMM_K_' + str(K), 'fold_' + str(kk), 'iter_' + str(iter))
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            launch_glm_hmm_job(inpt,
                               y,
                               session,
                               mask,
                               session_fold_lookup_table,
                               K,
                               D,
                               C,
                               N_em_iters,
                               transition_alpha,
                               prior_sigma,
                               kk,
                               iter,
                               global_fit,
                               init_param_file,
                               save_directory)
