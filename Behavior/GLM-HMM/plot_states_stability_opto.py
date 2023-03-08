import json
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, pearsonr
import seaborn as sns
from sklearn.metrics import jaccard_score
from matplotlib.colors import ListedColormap
from serotonin_functions import paths, load_trials, plot_psychometric, figure_style, load_subjects
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, load_animal_list, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from one.api import ONE
one = ONE()
np.random.seed(41)

# Settings
N_STATES = 4
MERGE_STATES = False # merge states 2 and 3

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
animal_list = load_animal_list(join(data_dir, 'animal_list.npz'))

# Exclude animals that do not have a previous action state
#animal_list = animal_list[~np.isin(animal_list, ['ZFM-04301'])]

colors, dpi = figure_style()
state_corr = pd.DataFrame()
state_block, state_probe = pd.DataFrame(), pd.DataFrame()
p_state_change, p_state_change, p_state_change_probe = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(animal_list):

    # Get genotype
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]    

    # Load in model
    results_dir = join(data_path, 'GLM-HMM', 'results', 'individual_fit', subject)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, N_STATES, results_dir, best_init_cvbt_dict)
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
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, N_STATES,
                                             range(N_STATES))
    states_max_posterior = np.argmax(posterior_probs, axis=1)

    if MERGE_STATES:
        states_max_posterior[states_max_posterior > 0] = 1
        K = N_STATES - 1
    else:
        K = N_STATES

    # Swap states 1 and 3 for some mice
    if subject in ['ZFM-05170', 'ZFM-04301']:
        states_max_posterior[states_max_posterior == 1] = 999
        states_max_posterior[states_max_posterior == 3] = 1
        states_max_posterior[states_max_posterior == 999] = 3
    
    # Loop over sessions
    trials = pd.DataFrame()
    for j, eid in enumerate(all_sessions):
        try:
            these_trials = load_trials(eid, laser_stimulation=True, patch_old_opto=False, one=one)
        except Exception as err:
            print(err)
            continue
        if np.where(session == eid)[0].shape[0] != these_trials.shape[0]:
            print(f'Session {eid} mismatch')
            continue
        these_trials['state'] = states_max_posterior[np.where(session == eid)[0]]
        these_trials['trial'] = these_trials.index.values
        these_trials['session'] = j
        for k in range(K):
            these_trials[f'state_{k+1}_probs'] = posterior_probs[np.where(session == eid)[0], k]
        trials = pd.concat((trials, these_trials), ignore_index=True)
    print(f'{subject}: {trials.shape[0]} trials')

    # Get state changes
    state_changes = np.where(np.abs(np.diff(trials['state'])) > 0)[0] + 1
    trials['state_change'] = np.zeros(trials.shape[0])
    trials.loc[state_changes, 'state_change'] = 1

    # Get probe trial triggered state switches
    trials['probe_trial'] = (trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1)
    opto_probe_df, opto_probe_bins_df = pd.DataFrame(), pd.DataFrame()
    all_probes = 0
    for s in np.unique(trials['session']):
        this_ses = trials[trials['session'] == s].reset_index(drop=True)
        opto_probe_ind = this_ses[this_ses['probe_trial']].index
        for b, trial_ind in enumerate(opto_probe_ind):
            
            # Single trials
            all_probes += 1
            these_switches = this_ses.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state_change'].values
            these_states = this_ses.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state'].values
            if trial_ind + TRIAL_BINS[0] < 0:
                trial_start = -trial_ind
            else:
                trial_start = TRIAL_BINS[0]
            if trial_ind + TRIAL_BINS[-1] > this_ses.shape[0]:
                trial_end = this_ses.shape[0] - trial_ind
            else:
                trial_end = TRIAL_BINS[-1]
            these_trials = np.concatenate((np.arange(trial_start, 0), np.arange(1, trial_end+1)))
            opto_probe_df = pd.concat((opto_probe_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial': these_trials, 'state': these_states,
                'probe_trial': all_probes})))
        
            # Binned trials
            these_switches = np.empty(len(TRIAL_BINS)-1)
            these_switches[:] = np.nan
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                if this_ses[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]].shape[0] == trial_bin_size:
                    these_switches[tt] = np.sum(this_ses.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1],
                                                             'state_change'].values)
            opto_probe_bins_df = pd.concat((opto_probe_bins_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial_bin': trial_bin_labels, 'probe_trial': all_probes,
                'trial_ind': np.arange(len(trial_bin_labels))})), ignore_index=True)
    opto_probe_bins_df['state_switch'] /= trial_bin_size

    # Remove probe trials
    trials.loc[(trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1), 'laser_stimulation'] = 0
    trials.loc[(trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0), 'laser_stimulation'] = 1
       
    # Get correlation of state with opto block
    jac_score = np.empty(K)
    for k in range(K):
        jac_score[k] = jaccard_score(trials['laser_stimulation'], trials['state'] == k)
    state_corr = pd.concat((state_corr, pd.DataFrame(data={
        'jaccard_score': jac_score, 'state': np.arange(1, K+1),
        'subject': subject, 'sert-cre': sert_cre})))
    
    # Get stimulation block triggered state switches
    trials['opto_block_switch'] = np.concatenate(([False], np.diff(trials['laser_stimulation']) != 0))
    opto_switch_df, opto_switch_bins_df = pd.DataFrame(), pd.DataFrame()
    all_blocks = 0
    for s in np.unique(trials['session']):
        this_ses = trials[trials['session'] == s].reset_index(drop=True)
        opto_block_switch_ind = this_ses[this_ses['opto_block_switch']].index

        for b, trial_ind in enumerate(opto_block_switch_ind):
            
            # Single trials
            all_blocks += 1
            these_switches = this_ses.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state_change'].values
            these_states = this_ses.loc[trial_ind+TRIAL_BINS[0]:trial_ind+TRIAL_BINS[-1]-1, 'state'].values
            if trial_ind + TRIAL_BINS[0] < 0:
                trial_start = -trial_ind
            else:
                trial_start = TRIAL_BINS[0]
            if trial_ind + TRIAL_BINS[-1] > this_ses.shape[0]:
                trial_end = this_ses.shape[0] - trial_ind
            else:
                trial_end = TRIAL_BINS[-1]
            these_trials = np.concatenate((np.arange(trial_start, 0), np.arange(1, trial_end+1)))
            opto_switch_df = pd.concat((opto_switch_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial': these_trials, 'state': these_states,
                'opto_switch': all_blocks, 'opto': this_ses.loc[trial_ind, 'laser_stimulation']})))
        
            # Binned trials
            these_switches = np.empty(len(TRIAL_BINS)-1)
            these_switches[:] = np.nan
            for tt, this_edge in enumerate(TRIAL_BINS[:-1]):
                if this_ses[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1]].shape[0] == trial_bin_size:
                    these_switches[tt] = np.sum(this_ses.loc[trial_ind+this_edge:trial_ind+TRIAL_BINS[tt+1],
                                                             'state_change'].values)
            opto_switch_bins_df = pd.concat((opto_switch_bins_df, pd.DataFrame(data={
                'state_switch': these_switches, 'trial_bin': trial_bin_labels,
                'opto_switch': all_blocks, 'trial_ind': np.arange(len(trial_bin_labels)),
                'opto': this_ses.loc[trial_ind, 'laser_stimulation']})), ignore_index=True)
    opto_switch_bins_df['state_switch'] /= trial_bin_size
    
    # Get P(state)
    state_switch_df = opto_switch_df[opto_switch_df['opto'] == 1].pivot(index='opto_switch', columns='trial', values='state')
    state_probe_df = opto_probe_df.pivot(index='probe_trial', columns='trial', values='state')

    # Add to overall dataframe
    this_state_switch = opto_switch_bins_df[opto_switch_bins_df['opto'] == 1].groupby('trial_ind').mean()['state_switch'] * 100
    p_state_change = pd.concat((p_state_change, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': sert_cre,
        'opto': 1})))
    this_state_switch = opto_switch_bins_df[opto_switch_bins_df['opto'] == 0].groupby('trial_ind').mean()['state_switch'] * 100
    p_state_change = pd.concat((p_state_change, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0],
        'opto': 0})))
    this_state_switch = opto_probe_bins_df.groupby('trial_ind').mean()['state_switch'] * 100
    p_state_change_probe = pd.concat((p_state_change_probe, pd.DataFrame(data={
        'p_change': this_state_switch,
        'p_change_bl': this_state_switch - np.mean(this_state_switch.values[:np.sum(trial_bin_labels < 0)]),
        'trial': trial_bin_labels, 'subject': subject,
        'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]})))
    this_state_df = pd.DataFrame()
    for k in range(K):
        this_p_state = np.mean(state_probe_df.values == k, axis=0)
        this_state_df = pd.concat((this_state_df, pd.DataFrame(data={
            'p_state': this_p_state,
            'p_state_bl': this_p_state - np.mean(this_p_state[state_probe_df.columns < 0]),
            'trial': state_probe_df.columns.values, 'subject': subject, 'state': k,
            'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]})))
    state_probe = pd.concat((state_probe, this_state_df))
    this_state_df = pd.DataFrame()
    for k in range(K):
        this_p_state = np.mean(state_switch_df.values == k, axis=0)
        this_state_df = pd.concat((this_state_df, pd.DataFrame(data={
            'p_state': this_p_state,
            'p_state_bl': this_p_state - np.mean(this_p_state[state_switch_df.columns < 0]),
            'trial': state_switch_df.columns.values, 'subject': subject, 'state': k,
            'sert-cre': subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]})))
    state_block = pd.concat((state_block, this_state_df))

    # Plot
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3.5), dpi=dpi)
    sns.lineplot(data=opto_probe_bins_df, x='trial_bin', y='state_switch', err_style='bars',
                 errorbar='se', color=colors['stim'], ax=ax1)
    ax1.set(ylabel='P(state switch)', xlabel='Trials since single stimulation', title=f'{subject}',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2))

    for k in range(K):
        ax2.plot(state_probe_df.columns.values, np.mean(state_probe_df.values == k, axis=0),
                 label=f'State {k+1}')
    ax2.set(ylabel='P(state)', xlabel='Trials since single stimulation',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2))
    ax2.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))

    sns.lineplot(data=opto_switch_bins_df, x='trial_bin', y='state_switch', hue='opto',
                 hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']],
                 errorbar='se', err_style='bars', ax=ax3)
    ax3.legend(frameon=False, prop={'size': 5}, title='opto')
    ax3.set(ylabel='P(state change)', xlabel='Trials since start of opto block')

    for k in range(K):
        ax4.plot(state_switch_df.columns.values, np.mean(state_switch_df.values == k, axis=0),
                 label=f'State {k+1}')
    ax4.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))
    ax4.set(ylabel='P(state)', xlabel='Trials since start of opto block',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2))

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(figure_dir, f'p_state_change_{subject}.jpg'), dpi=600)
    plt.close(f)

    # Plot this animal
    N_TRIALS = 1500
    if trials.shape[0] < N_TRIALS:
        N_TRIALS = trials.shape[0]
    f, ax1 = plt.subplots(1, 1, figsize=(5, 1.75), dpi=dpi)
    plt_states = ax1.imshow(np.array(trials.loc[:N_TRIALS-1, 'state'])[None, :],
                            aspect='auto', cmap=ListedColormap(sns.color_palette('Set1', K)),
                            alpha=0.7, extent=(1, N_TRIALS, -0.2, 1.2))
    ax1.plot(np.arange(1, N_TRIALS+1), trials.loc[:N_TRIALS-1, 'laser_stimulation'], color='k')
    ax1.set(xlabel='Trials', title=f'{subject}')
    cbar = f.colorbar(plt_states)
    cbar.set_ticks(np.arange(0, K))
    cbar.set_ticklabels(np.arange(1, K+1))
    plt.close(f)

# %% Do statistics

p_block, p_probe = np.empty(trial_bin_labels.shape[0]), np.empty(trial_bin_labels.shape[0])
for t, trial in enumerate(trial_bin_labels):
    p_probe[t] = ttest_1samp(p_state_change_probe.loc[(p_state_change_probe['trial'] == trial)
                                                      & (p_state_change_probe['sert-cre'] == 1),
                                                      'p_change_bl'], 0)[1]
    
    p_block[t] = ttest_ind(
        p_state_change.loc[(p_state_change['trial'] == trial) & (p_state_change['sert-cre'] == 1)
                           & (p_state_change['opto'] == 1), 'p_change_bl'],
        p_state_change.loc[(p_state_change['trial'] == trial) & (p_state_change['sert-cre'] == 1)
                           & (p_state_change['opto'] == 0), 'p_change_bl'])[1]
    
print(f'Probe {(p_probe < 0.05).astype(int)}')
print(f'Block {(p_block < 0.05).astype(int)}')


# %%
state_cols = [colors['state_1'], colors['state_2'], colors['state_3'], colors['state_4'], colors['state_5']]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.75), dpi=dpi)

ax1.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_state_change_probe[p_state_change_probe['sert-cre'] == 1],
             x='trial', y='p_change_bl', errorbar='se', color=colors['stim'],
             err_style='bars', ax=ax1)
#ax1.text(12, -0.75, f'n = {len(np.unique(p_state_change_probe.loc[p_state_change_probe["sert-cre"] == 1, "subject"]))} mice')
ax1.set(ylabel='P(state change) [%]', xlabel='Trials since single stimulation',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
        yticks=[-1, 0, 1, 2],
        title=f'n = {len(np.unique(p_state_change_probe.loc[p_state_change_probe["sert-cre"] == 1, "subject"]))} mice')

ax2.plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
sns.lineplot(data=p_state_change[p_state_change['sert-cre'] == 1], x='trial',
             y='p_change_bl', errorbar='se', hue='opto', err_style='bars',
             hue_order=[0, 1], palette=[colors['no-stim'], colors['stim']], ax=ax2)
ax2.set(ylabel='P(state change)', xlabel='Trials since start of stim. block',
        xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2))
handles, labels = ax2.get_legend_handles_labels()
labels = ['No stim block', 'Stim block']
ax2.legend(handles, labels, frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'state_change_prob_K{K}.jpg'), dpi=600)

# %%

f, axs = plt.subplots(1, K, figsize=(1.75*K, 1.75), dpi=dpi)
for k in range(K):
    axs[k].plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
    sns.lineplot(data=state_probe[(state_probe['sert-cre'] == 1) & (state_probe['state'] == k)],
                 x='trial', y='p_state_bl', errorbar='se', err_kws={'lw': 0}, color=state_cols[k],
                 ax=axs[k])
    axs[k].set(ylabel='P(state)', xlabel='Trials since single stimulation', title=f'State {k+1}',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
            ylim=[-0.05, 0.075])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'states_probe_{K}.jpg'), dpi=600)

# %%

f, axs = plt.subplots(1, K, figsize=(1.75*K, 1.75), dpi=dpi)
for k in range(K):
    axs[k].plot([TRIAL_BINS[0], TRIAL_BINS[-1]], [0, 0], ls='--', color='grey')
    sns.lineplot(data=state_block[(state_block['sert-cre'] == 1) & (state_block['state'] == k)],
                 x='trial', y='p_state_bl', errorbar='se', err_kws={'lw': 0}, color=state_cols[k],
                 ax=axs[k])
    axs[k].set(ylabel='P(state)', xlabel='Trials since start of stim. block', title=f'State {k+1}',
            xticks=np.arange(TRIAL_BINS[0], TRIAL_BINS[-1]+1, trial_bin_size*2),
            ylim=[-0.05, 0.075])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(figure_dir, f'states_block_{K}.jpg'), dpi=600)

# %%

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
sns.barplot(x='state', y='jaccard_score', data=state_corr[state_corr['sert-cre'] == 1], palette=state_cols[:K])
ax1.set(ylabel='State - opto similarity', xlabel='State')

sns.despine(trim=True)
plt.tight_layout()

