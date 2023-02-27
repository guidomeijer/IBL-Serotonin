#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:00:30 2022
By: Guido Meijer
"""

import json
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from serotonin_functions import paths
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, load_animal_list, \
    get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, get_prob_right

# Settings
K = 5

# Paths
data_path = '/home/guido/Data/5HT/'
figure_path = '/home/guido/Figures/5HT/'
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM', f'{K}_states')

# Load in data
animal_list = load_animal_list(join(data_dir, 'animal_list.npz'))
for animal in animal_list:
    results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', animal)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 results_dir,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Get pred acc:
    cols = ['#999999', '#984ea3', '#e41a1c', '#dede00']
    pred_acc_arr = load_cv_arr(join(results_dir, "predictive_accuracy_mat.npz"))
    pred_acc_arr_for_plotting = pred_acc_arr.copy()


    # %% ========== FIG 2c ==========
    fig = plt.figure(figsize=(2, 1.6), dpi=300)
    plt.subplots_adjust(left=0.3, bottom=0.3, right=0.9, top=0.9)
    plt.plot(np.arange(pred_acc_arr_for_plotting.shape[0]) + 1,
             np.mean(pred_acc_arr_for_plotting, axis=1),
             '-o',
             color=cols[1],
             zorder=0,
             alpha=1,
             lw=1.5,
             markersize=4)
    plt.gca().set(xticks=np.arange(pred_acc_arr_for_plotting.shape[0]) + 1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylabel("predictive acc. (%)", fontsize=10)
    plt.xlabel("# states", fontsize=10)
    plt.title(f'{animal}', fontsize=10)
    plt.tight_layout()
    fig.savefig(join(figure_dir, f'pred_acc{animal}.jpg'), dpi=600)

    # %% =========== Fig 2d =============
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 results_dir,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    transition_matrix = np.exp(hmm_params[1][0])

    fig = plt.figure(figsize=(2, 2), dpi=300)
    plt.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.95)
    plt.imshow(transition_matrix, vmin=-0.8, vmax=1, cmap='bone')
    if K < 4:
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                text = plt.text(j,
                                i,
                                str(np.around(transition_matrix[i, j],
                                              decimals=2)),
                                ha="center",
                                va="center",
                                color="k",
                                fontsize=10)
    plt.xlim(-0.5, K - 0.5)
    plt.xticks(range(0, K),
               ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
               fontsize=10)
    plt.yticks(range(0, K),
               ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
               fontsize=10)
    plt.ylim(K - 0.5, -0.5)
    plt.ylabel("state t-1", fontsize=10)
    plt.xlabel("state t", fontsize=10)
    plt.title(f'{animal}', fontsize=10)
    plt.tight_layout()
    fig.savefig(join(figure_dir, f'transition_mat_{animal}.jpg'), dpi=600)

    # %% =========== Fig 2e =============
    weight_vectors = -hmm_params[2]

    cols = [
        '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ]
    fig = plt.figure(figsize=(2.7, 2.5), dpi=300)
    plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
    M = weight_vectors.shape[2] - 1
    for k in range(K):
        plt.plot(range(M + 1),
                 weight_vectors[k][0][[0, 3, 1, 2]],
                 marker='o',
                 label="state " + str(k + 1),
                 color=cols[k],
                 lw=1,
                 alpha=0.7)
    plt.yticks([-2.5, 0, 2.5, 5], fontsize=10)
    plt.xticks(
        [0, 1, 2, 3],
        ['stimulus', 'bias', 'prev. \nchoice', 'win-stay-\nlose-switch'],
        fontsize=10,
        rotation=45)
    plt.ylabel("GLM weight", fontsize=10)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--", lw=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title(f'{animal}', fontsize=10)
    fig.savefig(join(figure_dir, f'weights_{animal}.jpg'), dpi=600)

    # %%
    alpha_val = 2
    sigma_val = 2

    fig = plt.figure(figsize=(4.6, 2), facecolor='w', edgecolor='k', dpi=300)
    plt.subplots_adjust(left=0.13, bottom=0.23, right=0.9, top=0.8)

    inpt, y, session = load_data(join(data_dir, animal + '_processed.npz'))
    unnormalized_inpt, _, _ = load_data(join(data_dir, animal + '_unnormalized.npz'))

    # Create masks for violation trials
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

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
    weight_vectors = hmm_params[2]

    posterior_probs = get_marginal_posterior(inputs, datas, masks,
                                             hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    cols = [
        '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ]
    for k in range(K):
        plt.subplot(1, K, k+1)
        # USE GLM WEIGHTS TO GET PROB RIGHT
        stim_vals, prob_right_max = get_prob_right(-weight_vectors, inpt, k, 1,
                                                   1)
        _, prob_right_min = get_prob_right(-weight_vectors, inpt, k, -1, -1)
        plt.plot(stim_vals,
                 prob_right_max,
                 '-',
                 color=cols[k],
                 alpha=1,
                 lw=1,
                 zorder=5)  # went R and was rewarded on previous trial
        plt.plot(stim_vals,
                 get_prob_right(-weight_vectors, inpt, k, -1, 1)[1],
                 '--',
                 color=cols[k],
                 alpha=0.5,
                 lw=1)  # went L and was not rewarded on previous trial
        plt.plot(stim_vals,
                 get_prob_right(-weight_vectors, inpt, k, 1, -1)[1],
                 '-',
                 color=cols[k],
                 alpha=0.5,
                 lw=1,
                 markersize=3)  # went R and was not rewarded on previous trial
        plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1,
                 lw=1)  # went L and was rewarded on previous trial
        plt.xticks([min(stim_vals), 0, max(stim_vals)],
                   labels=['', '', ''],
                   fontsize=10)
        plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
        plt.ylabel('')
        plt.xlabel('')
        plt.title(f"state {k}", fontsize=10, color=cols[k])

        if k == 0:
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['-100', '0', '100'],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
            plt.ylabel('p("R")', fontsize=10)
            plt.xlabel('stimulus', fontsize=10)
        if k == 1:
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
        if k == 2:
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)

        plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((-0.01, 1.01))
    plt.suptitle(f'{animal}', fontsize=10)
    #plt.tight_layout()
    fig.savefig(join(figure_dir, f'psy_curves_{animal}.jpg'), dpi=600)
