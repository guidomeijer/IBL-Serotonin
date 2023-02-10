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

N_STATES = {'ZFM-02181': 3, 'ZFM-02600': 3, 'ZFM-02601': 3, 'ZFM-03324': 5, 'ZFM-04080': 3,
            'ZFM-04122': 3, 'ZFM-03324': 5, 'ZFM-03329': 4, 'ZFM-03331': 3,
            'ZFM-04083': 4, 'ZFM-04300': 5, 'ZFM-04811': 5}
"""

n_states = 3
N_STATES = {'ZFM-02600': n_states, 'ZFM-02601': n_states, 'ZFM-03324': n_states,
            'ZFM-04080': n_states, 'ZFM-04122': n_states, 'ZFM-03321': n_states,
            'ZFM-03324': n_states, 'ZFM-03329': n_states, 'ZFM-03331': n_states,
            'ZFM-04083': n_states, 'ZFM-04300': n_states, 'ZFM-04811': n_states}
"""
#N_STATES = {'ZFM-03331': 3}
TRIAL_BINS = np.arange(-10, 31, 5)
trial_bin_size = np.unique(np.diff(TRIAL_BINS))[0]
#trial_bin_labels = [f'{i}-{i+trial_bin_size}' for i in TRIAL_BINS[:-1]]
trial_bin_labels = TRIAL_BINS[:-1] + (np.diff(TRIAL_BINS) / 2)

# Paths
figure_path, data_path = paths()
data_dir = join(data_path, 'GLM-HMM', 'data_by_animal')
figure_dir = join(figure_path, 'Behavior', 'GLM-HMM')

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()

plot_colors, dpi = figure_style()
state_change = pd.DataFrame()
p_state_change, p_state_change_bins, p_state_change_probe = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(list(N_STATES.keys())):

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

    """
    # Get probe trial triggered state switches
    trials['probe_trial'] = (trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1)
    opto_probe_ind = np.where(trials['probe_trial'])[0]
    opto_probe_df = pd.DataFrame()
    for b, trial_ind in enumerate(opto_probe_ind):
        these_switches = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state_change'].values
        these_states = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state'].values
        these_trials = np.concatenate((np.arange(TRIAL_BINS[0], 0), np.arange(1, TRIAL_BINS[-1]+1)))
        if (trial_ind + TRIAL_BINS[-1] < trials.shape[0]) & (trial_ind + TRIAL_BINS[0] > 0):
            opto_probe_df = pd.concat((opto_probe_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial': these_trials, 'state': these_states,
                'probe_trial': b})))


            these_switches = np.empty(len(TRIAL_BINS)-1)
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                these_switches[tt] = np.sum(trials.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]-1,
                                                       'state_change'].values)
            opto_probe_df = pd.concat((opto_probe_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial': trial_bin_labels, 'probe_trial': b})))
    """

    # Get stimulation block change triggered state switches
    trials['probe_trial'] = (trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1)
    opto_probe_ind = np.where(trials['probe_trial'])[0]
    opto_probe_df = pd.DataFrame()

    for b, trial_ind in enumerate(opto_probe_ind):

        if (trial_ind + TRIAL_BINS[-1] < trials.shape[0]) & (trial_ind + TRIAL_BINS[0] > 0):
            these_switches = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state_change'].values
            these_states = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state'].values
            these_trials = np.concatenate((np.arange(TRIAL_BINS[0], 0), np.arange(1, TRIAL_BINS[-1]+1)))
            opto_switch_df = pd.concat((opto_probe_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial': these_trials, 'state': these_states, 'opto_switch': b})))

            these_switches = np.empty(len(TRIAL_BINS)-1)
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                these_switches[tt] = np.sum(trials.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]-1,
                                                       'state_change'].values)
            opto_probe_df = pd.concat((opto_probe_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial_bin': trial_bin_labels, 'probe_trial': b,
                'trial_ind': np.arange(len(trial_bin_labels))})))

    # Remove probe trials
    trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'laser_stimulation'] = 1

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
    opto_switch_df, opto_switch_bins_df = pd.DataFrame(), pd.DataFrame()
    for b, trial_ind in enumerate(opto_block_switch_ind):

        if (trial_ind + TRIAL_BINS[-1] < trials.shape[0]) & (trial_ind + TRIAL_BINS[0] > 0):
            these_switches = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state_change'].values
            these_states = trials.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state'].values
            these_trials = np.concatenate((np.arange(TRIAL_BINS[0], 0), np.arange(1, TRIAL_BINS[-1]+1)))
            opto_switch_df = pd.concat((opto_switch_df, pd.DataFrame(data={
                'state_switch': these_switches, 'opto': trials.loc[trial_ind, 'laser_stimulation'],
                'trial': these_trials, 'state': these_states, 'opto_switch': b})))

            these_switches = np.empty(len(TRIAL_BINS)-1)
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                these_switches[tt] = np.sum(trials.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]-1,
                                                       'state_change'].values)
            opto_switch_bins_df = pd.concat((opto_switch_bins_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial_bin': trial_bin_labels, 'probe_trial': b,
                'trial_ind': np.arange(len(trial_bin_labels)),
                'opto': trials.loc[trial_ind, 'laser_stimulation']})))
    opto_switch_bins_df['state_switch'] /= trial_bin_size

    # Get P(state)
    state_switch_df = opto_switch_df[opto_switch_df['opto'] == 1].pivot(index='opto_switch', columns='trial', values='state')
    #state_probe_df = opto_probe_df.pivot(index='probe_trial', columns='trial', values='state')

    # Add to overall dataframe
    this_state_switch = opto_switch_df[opto_switch_df['opto'] == 1].groupby('trial').mean()['state_switch']
    p_state_change = pd.concat((p_state_change, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch[this_state_switch.index < 0]),
        'trial': np.unique(opto_switch_df['trial']), 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0],
        'opto': 'block'})))
    """
    this_state_switch = opto_probe_df.groupby('trial').mean()['state_switch']
    p_state_change = pd.concat((p_state_change, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch[this_state_switch.index < 0]),
        'trial': np.unique(opto_switch_df['trial']), 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0],
        'opto': 'probe'})))
    """
    this_state_switch = opto_switch_bins_df[opto_switch_bins_df['opto'] == 1].groupby('trial_ind').mean()['state_switch']
    p_state_change_bins = pd.concat((p_state_change_bins, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:3]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0],
        'opto': 1})))
    this_state_switch = opto_switch_bins_df[opto_switch_bins_df['opto'] == 0].groupby('trial_ind').mean()['state_switch']
    p_state_change_bins = pd.concat((p_state_change_bins, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:3]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0],
        'opto': 0})))

    this_state_switch = opto_probe_df.groupby('trial_ind').mean()['state_switch']
    p_state_change_probe = pd.concat((p_state_change_probe, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:3]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]})))

    # Plot
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=dpi)
    sns.lineplot(data=opto_switch_df[opto_switch_df['opto'] == 1], x='trial', y='state_switch',
                 errorbar='se', ax=ax1)
    ax1.set(ylabel='P(state change)', xlabel='Trials since start of opto block',
            title=f'{subject}')

    for k in range(K):
        ax2.plot(state_switch_df.columns.values, np.mean(state_switch_df.values == k, axis=0),
                 label=f'State {k+1}')
    ax2.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))
    ax2.set(ylabel='P(state)', xlabel='Trials since start of opto block',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, 5))

    sns.lineplot(data=opto_switch_bins_df, x='trial_bin', hue='opto',
                 y='state_switch', errorbar='se', err_style='bars', ax=ax3)
    ax3.set(ylabel='P(state change)', xlabel='Trials since start of opto block',
            title=f'{subject}')

    #sns.lineplot(data=opto_probe_df, x='trial', y='state_switch', errorbar='se',
    #             ax=ax4)
    #ax4.set(ylabel='P(state change)', xlabel='Trials since probe trial',
    #        title='Probe trials')

    #for k in range(K):
    #    ax5.plot(state_probe_df.columns.values, np.mean(state_probe_df.values == k, axis=0),
    #             label=f'State {k+1}')
    #ax5.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))
    #ax5.set(ylabel='P(state)', xlabel='Trials since probe trial',
    #        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, 5))

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(figure_dir, f'p_state_change_{subject}.jpg'), dpi=600)

    # Add to df
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0].astype(int)
    state_change = pd.concat((state_change, pd.DataFrame(index=[state_change.shape[0]+1], data={
        'subject': subject, 'sert_cre': sert_cre, 'state_probs': [state_probs],
        'state_probs_opto': [state_probs_opto], 'state_probs_no_opto': [state_probs_no_opto],
        'state_change_prob': state_change_prob, 'state_change_opto_prob': state_change_opto_prob,
        'state_change_no_opto_prob': state_change_no_opto_prob})))

    # Plot this animal
    N_TRIALS = 1500
    if trials.shape[0] < N_TRIALS:
        N_TRIALS = trials.shape[0]
    f, ax1 = plt.subplots(1, 1, figsize=(5, 1.75), dpi=dpi)
    plt_states = ax1.imshow(trials.loc[:N_TRIALS-1, 'state'][None, :],
                            aspect='auto', cmap=ListedColormap(sns.color_palette('Set1', K)),
                            alpha=0.7, extent=(1, N_TRIALS, -0.2, 1.2))
    ax1.plot(np.arange(1, N_TRIALS+1), trials.loc[:N_TRIALS-1, 'laser_stimulation'], color='k')
    ax1.set(xlabel='Trials', title=f'{subject}')
    cbar = f.colorbar(plt_states)
    cbar.set_ticks(np.arange(0, K))
    cbar.set_ticklabels(np.arange(1, K+1))

"""
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
"""

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

ax1.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
#ax1.plot([0, 0], [-0.05, 0.075], ls='--', color='grey')
sns.lineplot(data=p_state_change_bins[p_state_change_bins['sert-cre'] == 1], x='trial',
             y='p_change_bl', errorbar='se', hue='opto', err_style='bars',
             hue_order=[0, 1], palette=[plot_colors['no-stim'], plot_colors['stim']], ax=ax1)
ax1.set(ylabel='P(state change)', xlabel='Trials since start of stimulation block',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2))
#ax1.set_xticks(ax1.get_xticks())
#ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

handles, labels = ax1.get_legend_handles_labels()
labels = ['No stim', '5-HT stim']
ax1.legend(handles, labels, frameon=False, prop={'size': 5}, loc='upper left')


ax2.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
#ax2.plot([0, 0], [-0.1, 0.25], ls='--', color='grey')
sns.lineplot(data=p_state_change_probe[p_state_change_probe['sert-cre'] == 1],
             x='trial', y='p_change_bl', errorbar='se', color=plot_colors['stim'],
             err_style='bars', ax=ax2)
ax2.set(ylabel='P(state change)', xlabel='Trials since single stimulation',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
        yticks=[-0.1, 0, 0.1, 0.2, 0.3])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, 'state_change_prob.jpg'), dpi=600)

