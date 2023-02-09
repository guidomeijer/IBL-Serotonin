#  Fit GLM-HMM to data from all IBL animals together.  These fits will be
#  used to initialize the models for individual animals
import os
import sys
from serotonin_functions import paths
import autograd.numpy as np
from os.path import join
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
        load_animal_list, load_data, create_violation_mask, \
        launch_glm_hmm_job

# Settings
Ks = [2, 3, 4, 5]  # Number of latent states
prior_sigma = 2
transition_alpha = 2
D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations
num_folds = 5
z = 0
global_fit = False
N_initializations = 20  # Number of times to initialize the fit

# Paths
_, data_path = paths()
global_data_dir = join(data_path, 'GLM-HMM')
data_dir = join(global_data_dir, 'data_by_animal')
results_dir = join(global_data_dir, 'results', 'individual_fit')

animal_list = load_animal_list(join(data_dir, 'animal_list.npz'))
for i, animal in enumerate(animal_list):
    print(animal)
    animal_file = join(data_dir, animal + '_processed.npz')
    session_fold_lookup_table = load_session_fold_lookup(join(
        data_dir, animal + '_session_fold_lookup.npz'))

    inpt, y, session = load_data(animal_file)
    #  append a column of ones to inpt to represent the bias covariate:
    inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
    y = y.astype('int')

    overall_dir = join(results_dir, animal)

    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])

    # Loop over different models
    for K in Ks:
        init_param_file = join(global_data_dir, 'best_params', 'best_params_K_' + str(K) + '.npz')

        # Loop over folds and initializations
        for fold in range(num_folds):
            for iter in range(N_initializations):
                # create save directory for this initialization/fold combination:
                save_directory = join(overall_dir, 'GLM_HMM_K_' + str(K), 'fold_' + str(fold), 'iter_' + str(iter))
                if os.path.exists(save_directory):
                    continue
                else:
                    os.makedirs(save_directory)
                    launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table,
                                       K, D, C, N_em_iters, transition_alpha, prior_sigma,
                                       fold, iter, global_fit, init_param_file,
                                       save_directory)
