import json
import os
import sys
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from serotonin_functions import paths, load_trials, plot_psychometric, figure_style, load_subjects
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()
np.random.seed(41)

K = 3
MERGE_STATES = True

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()
glmhmm_subjects = os.listdir(join(data_path, 'GLM-HMM', 'results', 'individual_fit/'))
glmhmm_subjects = [i for i in glmhmm_subjects if i in subjects['subject'].values]

state_change = pd.DataFrame()
for i, subject in enumerate(glmhmm_subjects):

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
    all_sessions = np.unique(session)
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(np.hstack((inpt, np.ones((len(inpt), 1)))),
                                                           y, mask, session)

    # Get posterior probability of states per trial
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    if MERGE_STATES:
        states_max_posterior[states_max_posterior == 2] = 1  # merge states 2 and 3

    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(np.unique(session)):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except Exception as err:
            print(err)
            continue
        if np.where(session == eid)[0].shape[0] != these_trials.shape[0]:
            print(f'Session {eid} mismatch')
            continue
        these_trials['state'] = states_max_posterior[np.where(session == eid)[0]]
        for k in range(K):
            these_trials[f'state_{k+1}_probs'] = posterior_probs[np.where(session == eid)[0], k]
        trials = pd.concat((trials, these_trials), ignore_index=True)

    # Get state changes
    state_changes = np.where(np.abs(np.diff(trials['state'])) > 0)[0] + 1
    trials['state_change'] = np.zeros(trials.shape[0])
    trials.loc[state_changes, 'state_change'] = 1

    # Get state change probability per trial
    state_change_prob = (trials['state_change'].sum() / trials.shape[0]) * 100
    state_change_opto_prob = (trials.loc[trials['laser_stimulation'] == 1, 'state_change'].sum()
                              / trials[trials['laser_stimulation'] == 1].shape[0]) * 100
    state_change_no_opto_prob = (trials.loc[trials['laser_stimulation'] == 0, 'state_change'].sum()
                                 / trials[trials['laser_stimulation'] == 0].shape[0]) * 100

    # Get mean state probability
    state_probs, state_probs_opto, state_probs_no_opto = [], [], []
    for k in range(K):
        state_probs.append(trials[f'state_{k+1}_probs'].mean())
        state_probs_opto.append(trials.loc[trials['laser_stimulation'] == 1, f'state_{k+1}_probs'].mean())
        state_probs_no_opto.append(trials.loc[trials['laser_stimulation'] == 0, f'state_{k+1}_probs'].mean())

    # Add to df
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0].astype(int)
    state_change = pd.concat((state_change, pd.DataFrame(index=[state_change.shape[0]+1], data={
        'subject': subject, 'sert_cre': sert_cre, 'state_probs': [state_probs],
        'state_probs_opto': [state_probs_opto], 'state_probs_no_opto': [state_probs_no_opto],
        'state_change_prob': state_change_prob, 'state_change_opto_prob': state_change_opto_prob,
        'state_change_no_opto_prob': state_change_no_opto_prob})))

# %% plot
plot_colors, dpi = figure_style()
colors = [plot_colors['wt'], plot_colors['sert']]
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi, sharey=False)

for i, subject in enumerate(state_change['subject']):
    ax1.plot([1, 2],
             [state_change.loc[(state_change['subject'] == subject), 'state_change_no_opto_prob'],
              state_change.loc[(state_change['subject'] == subject), 'state_change_opto_prob']],
             color = colors[state_change.loc[state_change['subject'] == subject, 'sert_cre'].unique()[0]],
             marker='o', ms=2)

handles, labels = ax1.get_legend_handles_labels()
labels = ['', 'WT', 'SERT']
ax1.legend(handles[:3], labels[:3], frameon=False, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, .5))
ax1.set(xlabel='', ylabel='State change probability',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'state_change_prob_{K}K.jpg'), dpi=600)
