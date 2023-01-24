#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:50:59 2022
By: Guido Meijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from serotonin_functions import paths, figure_style, high_level_regions
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Settings
CLUSTER_N = 7  # number of clusters
PCA_DIM = 10

# Initialize
tsne = TSNE(n_components=2, random_state=42)
pca = PCA(n_components=PCA_DIM, random_state=42)

# Get paths
f_path, save_path = paths(dropbox=True)
fig_path = join(f_path, 'PaperPassive', 'figure2')

# Load in data
psth_df = pd.read_pickle(join(save_path, 'psth.pickle'))
neuron_type = pd.read_csv(join(save_path, 'neuron_type.csv'))

# Merge neuron type with PSTH df
neuron_type = neuron_type.rename(columns={'cluster_id': 'neuron_id'})
psth_df = pd.merge(psth_df, neuron_type, on=['subject', 'probe', 'eid', 'pid', 'neuron_id'])

# Exclude undefined neuron types
psth_df = psth_df[psth_df['type'] != 'Und.']

# Get high level regions
psth_df['high_level_region'] = high_level_regions(psth_df['acronym'])
psth_df = psth_df[psth_df['high_level_region'] != 'root']

# Do dimensionality reduction on PSTHs
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
time_ax = psth_df['time'][0]
for i in range(all_psth.shape[0]):
    #all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])  # normalize
    #all_psth[i, :] = all_psth[i, :] - np.mean(all_psth[i, time_ax < 0])  # baseline subtract
    #all_psth[i, :] = all_psth[i, :] / np.mean(all_psth[i, time_ax < 0])  # divide over baseline
    all_psth[i, :] = all_psth[i, :] / (np.mean(all_psth[i, time_ax < 0]) + 0.1)  # divide over baseline + 0.1 spks/s (Steinmetz, 2019)

# PCA
dim_red_pca = pca.fit_transform(all_psth)

# t-SNE
dim_red_tsne = tsne.fit_transform(all_psth)
psth_df['tsne_1'] = dim_red_tsne[:, 0]
psth_df['tsne_2'] = dim_red_tsne[:, 1]

# Clustering
psth_clusters = KMeans(n_clusters=CLUSTER_N, random_state=42, n_init='auto').fit_predict(dim_red_pca)
psth_df['cluster'] = psth_clusters

# %% Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', ax=ax1, legend=None)
ax1.axis('off')

plt.tight_layout()
plt.savefig(join(fig_path, 'tsne_embedding.pdf'))
plt.savefig(join(fig_path, 'tsne_embedding.jpg'), dpi=600)

# %%
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='cluster', ax=ax1, legend=None,
                palette='tab10')
ax1.axis('off')

plt.tight_layout()
plt.savefig(join(fig_path, 'tsne_embedding_clusters.pdf'))
plt.savefig(join(fig_path, 'tsne_embedding_cluster.jpg'), dpi=600)


# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)

psth_df['log_fr'] = np.log10(psth_df['firing_rate'])
sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='log_fr', ax=ax1, legend=None,
                palette='plasma')
ax1.axis('off')

plt.tight_layout()
plt.savefig(join(fig_path, 'tsne_embedding_firing_rate.pdf'))
plt.savefig(join(fig_path, 'tsne_embedding_firing_rate.jpg'), dpi=600)

# %% This one is actually part of figure 3
f, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=dpi)

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='type', ax=ax1, hue_order=['RS', 'NS'],
                palette=[colors['RS'], colors['NS']])
ax1.legend(frameon=False, prop={'size': 6})
ax1.axis('off')


plt.tight_layout()
plt.savefig(join(f_path, 'PaperPassive', 'figure3', 'tsne_embedding_neuron_type.pdf'))

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot([0, 0], [0, 2], ls='--', color='grey', lw=0.75)
for i in np.unique(psth_clusters):
    these_psth = all_psth[psth_clusters == i]
    ax1.fill_between(time_ax, np.mean(these_psth, axis=0) - (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    np.mean(these_psth, axis=0) + (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    alpha=0.25)
    ax1.plot(time_ax, np.mean(these_psth, axis=0))
ax1.set(ylabel='Firing rate over baseline', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4],
        yticks=[0, 0.5, 1, 1.5, 2])
plt.tight_layout()
sns.despine(trim=True, offset=-2)

plt.savefig(join(fig_path, 'cluster_psth.pdf'))

# %%
all_regions = np.unique(psth_df['high_level_region'])

f, axs = plt.subplots(1, all_regions.shape[0], figsize=(5, 1.5), dpi=dpi)

for i, region in enumerate(all_regions):
    axs[i].pie(np.bincount(psth_df.loc[psth_df['high_level_region'] == region, 'cluster']),
            colors=sns.color_palette(), radius=1.3)
    axs[i].set(title=region)

plt.savefig(join(fig_path, 'pie_chart_regions.pdf'))
