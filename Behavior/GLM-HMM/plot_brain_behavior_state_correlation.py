# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:37:13 2023

@author: Guido
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os.path import join
from sklearn.metrics import jaccard_score
from stim_functions import load_subjects, query_ephys_sessions, paths, figure_style
from plotting_utils import (load_glmhmm_data, load_cv_arr, load_data, load_animal_list, 
                            get_file_name_for_best_model_fold, partition_data_by_session, 
                            create_violation_mask, get_marginal_posterior, get_was_correct)
from one.api import ONE
one = ONE()

# Settings
N_STATES = 4
BIN_SIZE = 0.1

# Get paths
glm_hmm_dir = 'C:\\Users\\guido\\Data\\5-HT'
fig_path, save_path = paths()

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()
animal_list = load_animal_list(join(glm_hmm_dir, 'GLM-HMM', 'data_by_animal', 'animal_list.npz')) 

# Get ephys sessions
rec = query_ephys_sessions(n_trials=400, one=one)

# Load in brain state results
state_df = pd.read_csv(join(save_path, f'state_task_{int(BIN_SIZE*1000)}msbins.csv'))

sim_states_df = pd.DataFrame()
for i, subject in enumerate(animal_list):

    # Get genotype
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]    

    # Load in model
    results_dir = join(glm_hmm_dir, 'GLM-HMM', 'results', 'individual_fit', subject)
    cv_file = join(results_dir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, N_STATES, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Load in session data
    inpt, y, session = load_data(join(glm_hmm_dir, 'GLM-HMM', 'data_by_animal', subject + '_processed.npz'))
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
    
    """
    # Swap states 1 and 3 for some mice
    if subject in ['ZFM-05170', 'ZFM-04301']:
        states_max_posterior[states_max_posterior == 1] = 999
        states_max_posterior[states_max_posterior == 3] = 1
        states_max_posterior[states_max_posterior == 999] = 3
    """
    
    # Loop over sessions that have both GLM-HMM and ephys
    ephys_sessions = all_sessions[np.isin(all_sessions, rec['eid'])]
    if len(ephys_sessions) == 0:
        continue
    for j, eid in enumerate(ephys_sessions):
        
        # Get behavioral states per trail for this session
        behav_states = states_max_posterior[np.where(session == eid)[0]]
        
        # Loop over Neuropixel insertions in this session
        for k, pid in enumerate(np.unique(state_df.loc[state_df['eid'] == eid, 'pid'])):
            
            # Loop over regions recorded with this probe
            for r, region in enumerate(np.unique(state_df.loc[state_df['pid'] == pid, 'region'])):
                
                # Loop over time bins
                for t, time_bin in enumerate(np.unique(state_df['time'])):
                    
                    # Get brain state per trial
                    brain_states = state_df.loc[(state_df['pid'] == pid)
                                                & (state_df['region'] == region)
                                                & (state_df['time'] == time_bin), 'state'].values
                    
                    # Calculate similarity between brain states and behavioral states
                    for s, behav_state in enumerate(np.unique(behav_states)):
                        sim_states = np.empty(len(np.unique(brain_states)))
                        for ss, brain_state in enumerate(np.unique(brain_states)):
                            sim_states[ss] = jaccard_score((brain_states == brain_state).astype(int),
                                                           (behav_states == behav_state).astype(int))
                    
                        # Add to dataframe
                        sim_states_df = pd.concat((sim_states_df, pd.DataFrame(
                            index=[sim_states_df.shape[0]+1], data={
                            'behav_state': behav_state+1, 'similarity': np.max(sim_states),
                            'time': time_bin, 'region': region, 'pid': pid, 'subject': subject,
                            'sert-cre': sert_cre})))

# %% Plot

colors, dpi = figure_style()

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True) 
for r, region in enumerate(np.unique(sim_states_df['region'])): 
    sns.barplot(data=sim_states_df[(sim_states_df['region'] == region) & (sim_states_df['time'] == 0.75)],
                x='behav_state', y='similarity', errorbar='se', ax=axs[r])
    axs[r].set(title=f'{region}', xlabel='Behavioral state', ylabel='')
axs[0].set(ylabel='Brain-behavioral state similarity')
sns.despine(trim=True)
plt.tight_layout()  
plt.savefig(join(fig_path, 'Extra plots', 'Behavior', 'brain_behavior_state.jpg'), dpi=600)

grouped_df = sim_states_df.groupby(['pid', 'region', 'time']).max().reset_index()

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)   
sns.lineplot(data=grouped_df, x='time', y='similarity', hue='region', errorbar='se', err_style='bars',
             ax=ax1)

sns.lineplot(data=grouped_df[grouped_df['region'] == 'Midbrain'], x='time', y='similarity',
             errorbar='se', err_style='bars', ax=ax2)

sns.despine(trim=True)
plt.tight_layout()    

# %%
grouped_df = sim_states_df[(sim_states_df['time'] == 0.75)
                           & (sim_states_df['behav_state'] == 4)].groupby(['pid', 'region']).mean().reset_index()
      
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)      
sns.barplot(data=grouped_df, x='region', y='similarity', errorbar='se', ax=ax1)    
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set(xlabel='')        

sns.despine(trim=True)
plt.tight_layout()    

    
    
    
    
    
    
    
    
    