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

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()
glmhmm_subjects = os.listdir(join(data_path, 'GLM-HMM', 'results', 'individual_fit/'))
glmhmm_subjects = [i for i in glmhmm_subjects if i in subjects['subject'].values]

bias_df = pd.DataFrame()
for i, subject in enumerate(glmhmm_subjects):
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0].astype(int)

    # Load in model
    results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', subject)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, 3, results_dir, best_init_cvbt_dict)
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
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, 3, range(3))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
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
        trials = pd.concat((trials, these_trials), ignore_index=True)


    # Get bias strength 0% contrast trials
    for k in [0, 1]:
        bias_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                     & (trials['laser_stimulation'] == 0)
                                     & (trials['laser_probability'] == 0.25)
                                     & (trials['state'] == k)].mean()
                              - trials[(trials['probabilityLeft'] == 0.2)
                                       & (trials['laser_stimulation'] == 0)
                                       & (trials['laser_probability'] == 0.25)
                                       & (trials['state'] == k)].mean())['right_choice']
        bias_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                  & (trials['laser_stimulation'] == 1)
                                  & (trials['laser_probability'] == 0.75)
                                  & (trials['state'] == k)].mean()
                           - trials[(trials['probabilityLeft'] == 0.2)
                                    & (trials['laser_stimulation'] == 1)
                                    & (trials['laser_probability'] == 0.75)
                                    & (trials['state'] == k)].mean())['right_choice']

        bias_probe_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                        & (trials['laser_stimulation'] == 1)
                                        & (trials['laser_probability'] == 0.25)
                                        & (trials['state'] == k)].mean()
                                 - trials[(trials['probabilityLeft'] == 0.2)
                                          & (trials['laser_stimulation'] == 1)
                                          & (trials['laser_probability'] == 0.25)
                                          & (trials['state'] == k)].mean())['right_choice']
        bias_probe_no_stim = np.abs(trials[(trials['probabilityLeft'] == 0.8)
                                           & (trials['laser_stimulation'] == 0)
                                           & (trials['laser_probability'] == 0.75)
                                           & (trials['state'] == k)].mean()
                                    - trials[(trials['probabilityLeft'] == 0.2)
                                             & (trials['laser_stimulation'] == 0)
                                             & (trials['laser_probability'] == 0.75)
                                             & (trials['state'] == k)].mean())['right_choice']
        bias_df = pd.concat((bias_df, pd.DataFrame(index=[bias_df.shape[0]+1], data={
            'subject': subject, 'sert-cre': sert_cre, 'bias_no_opto': bias_no_stim,
            'bias_opto': bias_stim, 'bias_probe_opto': bias_probe_stim,
            'bias_probe_no_opto': bias_probe_no_stim, 'state': k})))

    # Plot
    colors, dpi = figure_style()
    f, axs = plt.subplots(1, 3, figsize=(3.5, 1.75), dpi=dpi, sharey=True)
    for k in [0, 1]:
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)
                                 & (trials['state'] == k)], ax=axs[k], color=colors['left'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)
                                 & (trials['state'] == k)], ax=axs[k],
                          color=colors['left'], linestyle='--')
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 0)
                                 & (trials['laser_probability'] != 0.75)
                                 & (trials['state'] == k)], ax=axs[k], color=colors['right'])
        plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                                 & (trials['laser_stimulation'] == 1)
                                 & (trials['laser_probability'] != 0.25)
                                 & (trials['state'] == k)], ax=axs[k],
                          color=colors['right'], linestyle='--')
        axs[k].text(-30, 0.8, f'State {k+1}')

    sns.despine(trim=True)
    plt.tight_layout()
    f.suptitle(f'{subject}, SERT: {sert_cre}', fontsize=7)
    plt.savefig(join(figure_dir, f'psy_states_{subject}.jpg'), dpi=600)

# %% Plot
colors, dpi = figure_style()
sert_colors = [colors['wt'], colors['sert']]
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.5, 3.5), dpi=dpi)

for i, subject in enumerate(bias_df['subject']):
    ax1.plot([1, 2], [bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 0), 'bias_no_opto'],
                      bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 0), 'bias_opto']],
             color = sert_colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax1.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Bias', ylim=[-0.1, 0.7], title='Engaged state')

for i, subject in enumerate(bias_df['subject']):
    ax2.plot([1, 2], [bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 1), 'bias_no_opto'],
                      bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 1), 'bias_opto']],
             color = sert_colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax2.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Bias', ylim=[-0.1, 0.7], title='Disengaged state')

for i, subject in enumerate(bias_df['subject']):
    ax3.plot([1, 2], [bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 0), 'bias_probe_no_opto'],
                      bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 0), 'bias_probe_opto']],
             color = sert_colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax3.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Bias', ylim=[-0.1, 0.7], title='Probe trials')

for i, subject in enumerate(bias_df['subject']):
    ax4.plot([1, 2], [bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 1), 'bias_probe_no_opto'],
                      bias_df.loc[(bias_df['subject'] == subject) & (bias_df['state'] == 1), 'bias_probe_opto']],
             color = sert_colors[bias_df.loc[bias_df['subject'] == subject, 'sert-cre'].values[0]], marker='o', ms=2)
ax4.set(xlabel='', xticks=[1, 2], xticklabels=['Non-stimulated\ntrials', 'Stimulated\ntrials'],
        ylabel='Bias', ylim=[-0.1, 0.7])

sns.despine(trim=True)
plt.tight_layout()

