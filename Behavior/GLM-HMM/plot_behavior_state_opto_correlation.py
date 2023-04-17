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
from brainbox.task.closed_loop import generate_pseudo_blocks
from sklearn.metrics import jaccard_score
from stim_functions import load_subjects, query_ephys_sessions, paths, figure_style, load_trials
from one.api import ONE
one = ONE()

# Settings
N_STATES = 4
BIN_SIZE = 0.1
N_NULL = 500

# Get paths
fig_path, save_path = paths()

# Get subjects for which GLM-HMM data is available
subjects = load_subjects()

# Get ephys sessions
rec = query_ephys_sessions(n_trials=400, one=one)

# Load in brain state results
state_df = pd.read_csv(join(save_path, f'state_task_{int(BIN_SIZE*1000)}msbins.csv'))

opto_state_df = pd.DataFrame()
for i in rec.index.values:
    if np.mod(i, 5) == 0:
        print(f'Recording {i} of {len(rec)}')

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]    

    # Load in trials
    trials = load_trials(eid, laser_stimulation=True, one=one)

    # Loop over regions recorded with this probe
    for r, region in enumerate(np.unique(state_df.loc[state_df['pid'] == pid, 'region'])):
        
        # Loop over time bins
        for t, time_bin in enumerate(np.unique(state_df['time'])):
            
            # Get brain state per trial
            brain_states = state_df.loc[(state_df['pid'] == pid)
                                        & (state_df['region'] == region)
                                        & (state_df['time'] == time_bin), 'state'].values
            
            # Calculate similarity between brain states and opto 
            sim_states = np.empty(len(np.unique(brain_states)))
            for s, brain_state in enumerate(np.unique(brain_states)):
                sim_states[s] = jaccard_score((brain_states == brain_state).astype(int),
                                               trials['laser_stimulation'].values)
        
            # Add to dataframe
            opto_state_df = pd.concat((opto_state_df, pd.DataFrame(
                index=[opto_state_df.shape[0]+1], data={
                'similarity': np.max(sim_states), 'time': time_bin, 'region': region, 'pid': pid,
                'subject': subject, 'sert-cre': sert_cre})))

# %% Plot
"""
colors, dpi = figure_style()

f, axs = plt.subplots(1, 7, figsize=(7, 1.75), dpi=dpi, sharey=True) 
for r, region in enumerate(np.unique(opto_state_df['region'])): 
    sns.barplot(data=sim_states_df[(opto_state_df['region'] == region) & (opto_state_df['time'] == 0.75)],
                x='behav_state', y='similarity', errorbar='se', ax=axs[r])
    axs[r].set(title=f'{region}', xlabel='Behavioral state', ylabel='')
axs[0].set(ylabel='Brain-behavioral state similarity')
sns.despine(trim=True)
plt.tight_layout()  


"""
grouped_df = opto_state_df.groupby(['pid', 'region', 'time']).max().reset_index()

colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2), dpi=dpi)   
sns.lineplot(data=grouped_df, x='time', y='similarity', hue='region', errorbar='se', err_style='bars',
             ax=ax1)
ax1.legend(bbox_to_anchor=[1,1])
ax1.set(xticks=[-1, 0, 1, 2, 3, 4], xlabel='Time (s)', ylabel='Brain state - opto similarity')

grouped_df = opto_state_df[opto_state_df['time'] == 0.75].groupby(['pid', 'region']).max().reset_index()
      
sns.barplot(data=grouped_df, x='region', y='similarity', errorbar='se', ax=ax2)    
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set(title='t = 0.75s', xlabel='', ylim=[0.2, 0.4])       

sns.despine(trim=True)
plt.tight_layout()   
plt.savefig(join(fig_path, 'Extra plots', 'State', 'state_opto_similarity.jpg'), dpi=600) 

 


    
    
    
    
    
    
    
    
    