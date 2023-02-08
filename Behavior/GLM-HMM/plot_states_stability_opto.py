import json
import os
import sys
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib.colors import ListedColormap
from serotonin_functions import paths, load_trials, plot_psychometric, figure_style, load_subjects
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()
np.random.seed(41)

# Settings
N_STATES = {'ZFM-01802': 4, 'ZFM-01864': 5, 'ZFM-01867': 3, 'ZFM-02181': 5, 'ZFM-02600': 4,
            'ZFM-02601': 3, 'ZFM-03324': 4, 'ZFM-04080': 4, 'ZFM-04122': 5}
TRIALS_BEFORE = 5
TRIALS_AFTER = 15
MERGE_STATES = False

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()
glmhmm_subjects = os.listdir(join(data_path, 'GLM-HMM', 'results', 'individual_fit/'))
glmhmm_subjects = [i for i in glmhmm_subjects if i in subjects['subject'].values]

plot_colors, dpi = figure_style()
state_change = pd.DataFrame()
for i, subject in enumerate(glmhmm_subjects):

    # Get number of states for this animal
    K = N_STATES[subject]

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

    # Remove probe trials
    trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'laser_stimulation'] = 1

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

    # Get stimulation block change triggered state switches
    trials['opto_block_switch'] = np.concatenate(([False], np.diff(trials['laser_stimulation']) != 0))
    opto_block_switch_ind = np.where(trials['opto_block_switch'])[0]
    opto_switch_df = pd.DataFrame()
    for b, trial_ind in enumerate(opto_block_switch_ind):
        these_switches = trials.loc[trial_ind-TRIALS_BEFORE:trial_ind+TRIALS_AFTER-1, 'state_change'].values
        these_trials = np.concatenate((np.arange(-TRIALS_BEFORE, 0), np.arange(1, TRIALS_AFTER+1)))
        if (trial_ind + TRIALS_AFTER < trials.shape[0]) & (trial_ind - TRIALS_BEFORE > 0):
            opto_switch_df = pd.concat((opto_switch_df, pd.DataFrame(data={
                'state_switch': these_switches, 'opto': trials.loc[trial_ind, 'laser_stimulation'],
                'trial': these_trials})))

    # Correlate states with opto blocks
    r_state = np.empty(K)
    for k in range(K):
        r_state[k] = pearsonr((trials['state'] == k).astype(int), trials['laser_stimulation'])[0]

    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
    sns.lineplot(data=opto_switch_df[opto_switch_df['opto'] == 1], x='trial', y='state_switch',
                 errorbar='se', ax=ax1)
    ax1.set(ylabel='P(state change)', xlabel='Trials since start of opto block',
            title=f'{subject}', xticks=np.arange(-TRIALS_BEFORE, TRIALS_AFTER+1, 5))

    ax2.bar(np.arange(1, K+1), r_state)
    ax2.set(ylabel='State opto correlation (r)', xlabel='State')

    sns.despine(trim=True)
    plt.tight_layout()



    # Add to df
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0].astype(int)
    state_change = pd.concat((state_change, pd.DataFrame(index=[state_change.shape[0]+1], data={
        'subject': subject, 'sert_cre': sert_cre, 'state_probs': [state_probs],
        'state_probs_opto': [state_probs_opto], 'state_probs_no_opto': [state_probs_no_opto],
        'state_change_prob': state_change_prob, 'state_change_opto_prob': state_change_opto_prob,
        'state_change_no_opto_prob': state_change_no_opto_prob})))

    # Plot this animal
    N_TRIALS = 1500
    f, ax1 = plt.subplots(1, 1, figsize=(5, 1.75), dpi=dpi)
    plt_states = ax1.imshow(trials.loc[:N_TRIALS-1, 'state'][None, :],
                            aspect='auto', cmap=ListedColormap(sns.color_palette('Set1', K)),
                            alpha=0.7, extent=(1, N_TRIALS, -0.2, 1.2))
    ax1.plot(np.arange(1, N_TRIALS+1), trials.loc[:N_TRIALS-1, 'laser_stimulation'], color='k')
    ax1.set(xlabel='Trials', title=f'{subject}')
    cbar = f.colorbar(plt_states)
    cbar.set_ticks(np.arange(0, K))
    cbar.set_ticklabels(np.arange(1, K+1))

# %% plot
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
ax1.set(xlabel='', ylabel='State change probability (%)',
        xticks=[1, 2], xticklabels=['No stim', 'Stim'])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'state_change_prob_{K}K.jpg'), dpi=600)
