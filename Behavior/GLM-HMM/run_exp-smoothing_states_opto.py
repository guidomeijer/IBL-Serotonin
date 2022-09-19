import json
import os
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon
from models.expSmoothing_prevAction_4lr import expSmoothing_prevAction_4lr as exp_prev_action
from serotonin_functions import (paths, figure_style, load_exp_smoothing_trials, load_subjects)
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()
np.random.seed(41)

# Settings
REMOVE_OLD_FIT = True
K = 2

# Get subjects
subjects = load_subjects()

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

# Get subjects for which GLM-HMM data is available
glmhmm_subjects = os.listdir(join(data_path, 'GLM-HMM', 'results', 'individual_fit/'))
glmhmm_subjects = [i for i in glmhmm_subjects if i in subjects['subject'].values]

results_df = pd.DataFrame()
for i, subject in enumerate(glmhmm_subjects):
    print(f'Starting {subject} ({i+1} of {len(glmhmm_subjects)})')

    # Load in model
    results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', subject)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Load in session data
    inpt, y, session = load_data(join(data_dir, subject + '_processed.npz'))
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(np.hstack((inpt, np.ones((len(inpt), 1)))),
                                                           y, mask, session)

    # Get posterior probability of states per trial
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    if K == 3:
        states_max_posterior[states_max_posterior == 2] = 1  # merge states 2 and 3

    # Get trial data
    all_sessions = np.unique(session)
    if all_sessions.shape[0] > 10:
        all_sessions = all_sessions[:10]
    actions, stimuli, stim_side, prob_left, stim_trials, session_uuids = load_exp_smoothing_trials(
        all_sessions, stimulated='block', one=one)

    # Create state vector in the shape of exp smoothing model input
    state_trials = np.zeros(stim_trials.shape)
    for j, this_eid in enumerate(session_uuids):
        state_trials[j, :np.sum(session == this_eid)] = states_max_posterior[np.where(session == this_eid)[0]]

    # Combine state and opto stimulation arrays into one
    opto_state = np.zeros(stim_trials.shape)
    opto_state[(stim_trials == 0) & (state_trials == 0)] = 0  # non-stimulated engaged
    opto_state[(stim_trials == 0) & (state_trials == 1)] = 1  # non-stimulated disengaged
    opto_state[(stim_trials == 1) & (state_trials == 0)] = 2  # stimulated engaged
    opto_state[(stim_trials == 1) & (state_trials == 1)] = 3  # stimulated disengaged
    opto_state = torch.tensor(opto_state.astype(int))

    # Fit model
    model = exp_prev_action(join(data_path, 'exp-smoothing/'), session_uuids, '%s_opto_state' % subject,
                            actions, stimuli, stim_side, opto_state)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_prevaction = model.get_parameters(parameter_type='posterior_mean')

    output_prevaction = model.compute_signal(signal=['prior', 'score'], act=actions, stim=stimuli, side=stim_side)
    priors_prevaction = output_prevaction['prior']
    results_df = pd.concat((results_df, pd.DataFrame(data={
        'tau_pa': 1/param_prevaction[:4], 'opto_stim': [0, 0, 1, 1], 'subject': subject,
        'state': ['engaged', 'disengaged', 'engaged', 'disengaged'],
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0].astype(int)})))

# %% Plot
plot_colors, dpi = figure_style()
colors = [plot_colors['wt'], plot_colors['sert']]
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.5, 3.5), dpi=dpi, sharey=False)

_, p = wilcoxon(results_df.loc[(results_df['sert-cre'] == 1) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 0), 'tau_pa'],
                results_df.loc[(results_df['sert-cre'] == 1) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 1), 'tau_pa'])

for i, subject in enumerate(results_df['subject']):
    ax1.plot([1, 2],
             [results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 0), 'tau_pa'],
              results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 1), 'tau_pa']],
             color = colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)
handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='Length of integration window (tau)', title='Engaged state',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 15])


for i, subject in enumerate(results_df['subject']):
    ax2.plot([1, 2],
             [results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'disengaged') & (results_df['opto_stim'] == 0), 'tau_pa'],
              results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'disengaged') & (results_df['opto_stim'] == 1), 'tau_pa']],
             color = colors[results_df.loc[results_df['subject'] == subject, 'sert-cre'].unique()[0]],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax2.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax2.set(xlabel='', ylabel='Length of integration window (tau)', title='Disengaged state',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'], ylim=[0, 15])


for i, subject in enumerate(results_df['subject']):
    ax3.plot([1, 2],
             [results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 1), 'tau_pa'],
              results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'disengaged') & (results_df['opto_stim'] == 1), 'tau_pa']],
             color = plot_colors['stim'],
             marker='o', ms=2)
    ax3.plot([1, 2],
             [results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'engaged') & (results_df['opto_stim'] == 0), 'tau_pa'],
              results_df.loc[(results_df['subject'] == subject) & (results_df['state'] == 'disengaged') & (results_df['opto_stim'] == 0), 'tau_pa']],
             color = plot_colors['no-stim'],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax3.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax3.set(xlabel='', ylabel='Length of integration window (tau)',
        xticks=[1, 2], xticklabels=['Engaged', 'Disengaged'], ylim=[0, 15])

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'Opto', 'No opto']
ax3.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'exp-smoothing_states_{K}K.jpg'), dpi=600)




