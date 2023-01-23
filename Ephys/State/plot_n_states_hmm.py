#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:27:38 2023
By: Guido Meijer
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from serotonin_functions import figure_style, paths, load_subjects
from os.path import join

# Get paths
fig_path, save_path = paths()

# Load in data
hmm_ll_df = pd.read_csv(join(save_path, 'HMM', 'hmm_log_likelihood.csv'))

# Convert ll
hmm_ll_df['xcorr'] = -2 * hmm_ll_df['log_likelihood']

# Normalize ll
n_states = np.unique(hmm_ll_df['n_states'])
for i in hmm_ll_df[hmm_ll_df['n_states'] == hmm_ll_df['n_states'].min()].index:
    hmm_ll_df.loc[i:i+len(n_states)-1, 'll_norm'] = (hmm_ll_df.loc[i:i+len(n_states)-1, 'xcorr']
                                                     / hmm_ll_df.loc[i, 'xcorr'])

# Average within mice first
ll_mean_df = hmm_ll_df.groupby(['subject', 'n_states', 'region']).mean(numeric_only=True).reset_index()

# Select only sert-cre mice
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    ll_mean_df.loc[ll_mean_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]
ll_mean_df = ll_mean_df[ll_mean_df['sert-cre'] == 1]

# Plot anesthesia
colors, dpi = figure_style()
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(7, 3), dpi=dpi)
sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Frontal'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax1, marker='o')
ax1.set(title='Frontal cortex', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Sensory'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax2, marker='o')
ax2.set(title='Sensory cortex', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Hippocampus'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax3, marker='o')
ax3.set(title='Hippocampus', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Striatum'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax4, marker='o')
ax4.set(title='Striatum', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Thalamus'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax5, marker='o')
ax5.set(title='Thalamus', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Midbrain'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax6, marker='o')
ax6.set(title='Midbrain', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Amygdala'], x='n_states', y='ll_norm',
             estimator=None, units='subject', ax=ax7, marker='o')
ax7.set(title='Amygdala', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

plt.tight_layout()
sns.despine(trim=True)

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(7, 3), dpi=dpi)
sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Frontal'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax1, marker='o')
ax1.set(title='Frontal cortex', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Sensory'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax2, marker='o')
ax2.set(title='Sensory cortex', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Hippocampus'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax3, marker='o')
ax3.set(title='Hippocampus', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Striatum'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax4, marker='o')
ax4.set(title='Striatum', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Thalamus'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax5, marker='o')
ax5.set(title='Thalamus', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Midbrain'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax6, marker='o')
ax6.set(title='Midbrain', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

sns.lineplot(data=ll_mean_df[ll_mean_df['region'] == 'Amygdala'], x='n_states', y='ll_norm',
             errorbar='se', ax=ax7, marker='o')
ax7.set(title='Amygdala', ylabel='Normalized log likelihood', xlabel='Number of states',
        xticks=np.arange(2, 21, 2))

plt.tight_layout()
sns.despine(trim=True)