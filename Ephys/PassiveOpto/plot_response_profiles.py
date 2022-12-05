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
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
pca = PCA(n_components=10, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

# Settings
cluster_n = 7  # number of clusters

# Get paths
f_path, save_path = paths()
fig_path = join(f_path, 'Ephys', 'ResponseProfiles')

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
# t-SNE
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
for i in range(all_psth.shape[0]):
    all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])  # normalize
dim_red_psth = tsne.fit_transform(all_psth)
psth_df['tsne_1'] = dim_red_psth[:, 0]
psth_df['tsne_2'] = dim_red_psth[:, 1]

# PCA
psth_pca = pca.fit_transform(all_psth)
psth_df['pca_1'] = psth_pca[:, 0]
psth_df['pca_2'] = psth_pca[:, 1]

# Determine optimal number of clusters
inertia_o = np.square((psth_pca - psth_pca.mean(axis=0))).sum()
inertia = np.empty(15)
for k in range(15):
    # fit k-means
    kmeans = KMeans(n_clusters=k+1, random_state=42).fit(psth_pca)
    inertia[k] = kmeans.inertia_

# Plot optimal cluster number
colors, dpi = figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
ax1.plot(np.arange(1, 16), inertia, marker='o')
ax1.set(ylabel='Inertia', xlabel='Number of clusters',
        xticks=[1, 5, 10, 15], yticks=[200, 700, 1200])
plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'kmeans_inertia.jpg'), dpi=600)

# Clustering
psth_clusters = KMeans(n_clusters=cluster_n, random_state=42).fit_predict(psth_pca)
psth_df['cluster'] = psth_clusters

# %% Plot

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(7, 1.75), dpi=300)
sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='cluster', ax=ax1, legend=None,
                palette='tab10')
ax1.axis('off')
ax1.set(title='Clustering')

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='type', ax=ax2, hue_order=['NS', 'RS'],
                palette=[colors['NS'], colors['RS']])
ax2.legend(frameon=False, prop={'size': 5})
ax2.axis('off')
ax2.set(title='Neuron type')

psth_df['log_fr'] = np.log10(psth_df['firing_rate'])
sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='log_fr', ax=ax3, legend=None,
                palette='plasma')
ax3.axis('off')
ax3.set(title='Firing rate')

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='high_level_region', ax=ax4,
                hue_order=np.unique(psth_df['high_level_region']),
                palette=[colors.get(key) for key in np.unique(psth_df['high_level_region'])])
ax4.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))
ax4.set(title='Region')
#ax2.legend(frameon=False, prop={'size': 5})
ax4.axis('off')

plt.tight_layout()
plt.savefig(join(fig_path, 'tsne_embedding.jpg'), dpi=600)

# %% Plot

colors, dpi = figure_style()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=300)
sns.scatterplot(data=psth_df, x='pca_1', y='pca_2', hue='type', ax=ax1, hue_order=['NS', 'RS'],
                palette=[colors['NS'], colors['RS']])
ax1.legend(frameon=False, prop={'size': 5})
ax1.axis('off')

sns.scatterplot(data=psth_df, x='pca_1', y='pca_2', hue='high_level_region', ax=ax2, legend=None)
#ax2.legend(frameon=False, prop={'size': 5})
ax2.axis('off')
plt.savefig(join(fig_path, 'pca_embedding.jpg'), dpi=600)

# %%
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
time_ax = psth_df['time'][0]
fs_psth = np.column_stack(psth_df.loc[psth_df['type'] == 'FS', 'peth'].to_numpy()).T
for i in range(fs_psth.shape[0]):
    fs_psth[i, :] = fs_psth[i, :] - np.mean(fs_psth[i, time_ax < 0])  # baseline subtract
ax1.fill_between(time_ax, np.mean(fs_psth, axis=0) - (np.std(fs_psth, axis=0) / np.sqrt(fs_psth.shape[0])),
                np.mean(fs_psth, axis=0) + (np.std(fs_psth, axis=0) / np.sqrt(fs_psth.shape[0])),
                alpha=0.25, color=colors['FS'])
ax1.plot(time_ax, np.mean(fs_psth, axis=0), color=colors['FS'], label='FS')

rs_psth = np.column_stack(psth_df.loc[psth_df['type'] == 'RS', 'peth'].to_numpy()).T
for i in range(rs_psth.shape[0]):
    rs_psth[i, :] = rs_psth[i, :] - np.mean(rs_psth[i, time_ax < 0])  # baseline subtract
ax1.fill_between(time_ax, np.mean(rs_psth, axis=0) - (np.std(rs_psth, axis=0) / np.sqrt(rs_psth.shape[0])),
                np.mean(rs_psth, axis=0) + (np.std(rs_psth, axis=0) / np.sqrt(rs_psth.shape[0])),
                alpha=0.25, color=colors['RS'])
ax1.plot(time_ax, np.mean(rs_psth, axis=0), color=colors['RS'], label='RS')

ax1.legend(frameon=False, prop={'size': 5})
ax1.set(ylabel='Baseline subtracted spks/s', xlabel='Time (s)', yticks=[-3, -2, -1, 0, 1])
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'RS_FS_mean_PSTH.jpg'), dpi=600)
"""
# %%
time_ax = psth_df['time'][0]
all_psth = np.column_stack(psth_df['peth'].to_numpy()).T
for i in range(all_psth.shape[0]):
    all_psth[i, :] = all_psth[i, :] - np.mean(all_psth[i, time_ax < 0])  # baseline subtract

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
for i in np.unique(psth_clusters):
    these_psth = all_psth[psth_clusters == i]
    ax1.fill_between(time_ax, np.mean(these_psth, axis=0) - (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    np.mean(these_psth, axis=0) + (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    alpha=0.25)
    ax1.plot(time_ax, np.mean(these_psth, axis=0), label='FS')
ax1.set(ylabel='Firing rate change (spks/s)', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'cluster_psth.jpg'), dpi=600)

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.pie(np.bincount(psth_df.loc[psth_df['type'] == 'FS', 'cluster']), colors=sns.color_palette())
ax1.set(title='FS')

ax2.pie(np.bincount(psth_df.loc[psth_df['type'] == 'RS', 'cluster']), colors=sns.color_palette())
ax2.set(title='RS')

plt.savefig(join(fig_path, 'pie_chart_clusters.jpg'), dpi=600)

# %%
all_regions = np.unique(psth_df['high_level_region'])

f, axs = plt.subplots(1, all_regions.shape[0], figsize=(7, 1.75), dpi=dpi)

for i, region in enumerate(all_regions):
    axs[i].pie(np.bincount(psth_df.loc[psth_df['high_level_region'] == region, 'cluster']),
            colors=sns.color_palette())
    axs[i].set(title=region)

plt.savefig(join(fig_path, 'pie_chart_regions.jpg'), dpi=600)