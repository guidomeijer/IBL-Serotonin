# Create panels a-c of Figure 3 of Ashwood et al. (2020)
import json
import os
import sys
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from serotonin_functions import paths
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct


animal = "ZFM-02600"
K = 3

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

# Get the file name corresponding to the best initialization for given K
# value
raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                             results_dir,
                                             best_init_cvbt_dict)
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
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])
y[np.where(y == -1), :] = 1
inputs, datas, train_masks = partition_data_by_session(
    np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
    session)

posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                         hmm_params, K, range(K))
states_max_posterior = np.argmax(posterior_probs, axis=1)

sess_to_plot = ["30d70f60-553b-4325-ae83-a35f23271c02", "56fc65b3-8e8b-4e65-a935-71cc75eb5e5c",
                "a98812f6-f5ac-4022-a17b-8163f26fa780"]

cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']
fig = plt.figure(figsize=(6, 4), dpi=400)
plt.subplots_adjust(wspace=0.2, hspace=0.9)
for i, sess in enumerate(sess_to_plot):
    plt.subplot(2, 3, i + 4)
    idx_session = np.where(session == sess)
    this_inpt, this_y = inpt[idx_session[0], :], y[idx_session[0], :]
    was_correct, idx_easy = get_was_correct(this_inpt, this_y)
    this_y = this_y[:, 0] + np.random.normal(0, 0.03, len(this_y[:, 0]))
    # plot choice, color by correct/incorrect:
    locs_correct = np.where(was_correct == 1)[0]
    locs_incorrect = np.where(was_correct == 0)[0]
    plt.plot(locs_correct, this_y[locs_correct], 'o', color='black',
             markersize=2, zorder=3, alpha=0.5)
    plt.plot(locs_incorrect, this_y[locs_incorrect], 'v', color='red',
             markersize=2, zorder=4, alpha=0.5)

    states_this_sess = states_max_posterior[idx_session[0]]
    state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
    for change_loc in state_change_locs:
        plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
    plt.ylim((-0.13, 1.13))
    if i == 0:
        plt.yticks([0, 1], ["L", "R"], fontsize=10)
    else:
        plt.yticks([0, 1], ["", ""], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title("example session " + str(i + 1), fontsize=10)
    plt.gca().set(xlim=[0, 200])
    if i == 0:
        plt.xlabel("trial #", fontsize=10)
        plt.ylabel("choice", fontsize=10)

for i, sess in enumerate(sess_to_plot):
    plt.subplot(2, 3, i + 1)
    idx_session = np.where(session == sess)
    this_inpt = inpt[idx_session[0], :]
    posterior_probs_this_session = posterior_probs[idx_session[0], :]
    # Plot trial structure for this session too:
    for k in range(K):
        plt.plot(posterior_probs_this_session[:, k],
                 label="State " + str(k + 1), lw=1,
                 color=cols[k])
    states_this_sess = states_max_posterior[idx_session[0]]
    state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
    for change_loc in state_change_locs:
        plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
    if i == 0:
        plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
    else:
        plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
    plt.ylim((-0.01, 1.01))
    plt.title("example session " + str(i + 1), fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set(xlim=[0, 200])
    if i == 0:
        plt.xlabel("trial #", fontsize=10)
        plt.ylabel("p(state)", fontsize=10)


fig.savefig(join(figure_dir, f'example_sessions_{animal}.pdf'))
