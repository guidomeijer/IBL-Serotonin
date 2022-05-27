# Create panels a-c of Figure 3 of Ashwood et al. (2020)
import json
import os
import sys
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from serotonin_functions import paths, load_trials, figure_style
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()

animal = "ZFM-02600"
K = 4  # Number of states

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', animal)
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

np.random.seed(41)

cv_file = join(results_dir, "cvbt_folds_model.npz")
cvbt_folds_model = load_cv_arr(cv_file)

with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
    best_init_cvbt_dict = json.load(f)

# Get the file name corresponding to the best initialization for given K value
raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
hmm_params, lls = load_glmhmm_data(raw_file)

# Save parameters for initializing individual fits
weight_vectors = hmm_params[2]
log_transition_matrix = hmm_params[1][0]
init_state_dist = hmm_params[0][0]

# Also get data for animal:
inpt, y, session = load_data(join(data_dir, animal + '_processed.npz'))
all_sessions = np.unique(session)

# Create mask:
# Identify violations for exclusion:
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
y[np.where(y == -1), :] = 1
inputs, datas, train_masks = partition_data_by_session(np.hstack((inpt, np.ones((len(inpt), 1)))),
                                                       y, mask, session)
posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K))
states_max_posterior = np.argmax(posterior_probs, axis=1)

# Loop over sessions
trials = pd.DataFrame()
for i, eid in enumerate(np.unique(session)):
    these_trials = load_trials(eid, laser_stimulation=True, one=one)
    these_trials['state'] = states_max_posterior[np.where(session == eid)[0]]
    trials = pd.concat((trials, these_trials), ignore_index=True)

# Get bias for stim and no stim per state
bias_stim, bias_no_stim = np.empty(K), np.empty(K)
for k in range(K):
    bias_no_stim[k] = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['state'] == k)].mean()
                          - trials[(trials['probabilityLeft'] == 0.2)
                                   & (trials['laser_stimulation'] == 0)
                                   & (trials['state'] == k)].mean())['right_choice']
    bias_stim[k] = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['state'] == k)].mean()
                          - trials[(trials['probabilityLeft'] == 0.2)
                                   & (trials['laser_stimulation'] == 1)
                                   & (trials['state'] == k)].mean())['right_choice']