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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
tsne = TSNE(n_components=2, random_state=42)

# Settings
CLUSTER_N = 7  # number of clusters

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
time_ax = psth_df['time'][0]
for i in range(all_psth.shape[0]):
    #all_psth[i, :] = all_psth[i, :] / np.max(all_psth[i, :])  # normalize
    #all_psth[i, :] = all_psth[i, :] - np.mean(all_psth[i, time_ax < 0])  # baseline subtract
    all_psth[i, :] = all_psth[i, :] / np.mean(all_psth[i, time_ax < 0])  # divide over baseline
    #all_psth[i, :] = all_psth[i, :] / (np.mean(all_psth[i, time_ax < 0] + 1))  # divide over baseline + 1 spks/s (Steinmetz, 2019)
dim_red_psth = tsne.fit_transform(all_psth)
psth_df['tsne_1'] = dim_red_psth[:, 0]
psth_df['tsne_2'] = dim_red_psth[:, 1]

# Determine optimal number of clusters
inertia_o = np.square((dim_red_psth - dim_red_psth.mean(axis=0))).sum()
inertia = np.empty(12)
silhouette_scores = np.empty(12)
for k in range(12):
    # fit k-means
    kmeans = KMeans(n_clusters=k+2, random_state=42, n_init='auto').fit(dim_red_psth)
    inertia[k] = kmeans.inertia_
    silhouette_scores[k] = silhouette_score(dim_red_psth, kmeans.labels_)

# Plot optimal cluster number
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.plot(np.arange(2, 14), inertia, marker='o')
ax1.set(ylabel='Inertia', xlabel='Number of clusters', xticks=[2, 8, 14])
ax2.plot(np.arange(2, 14), silhouette_scores, marker='o')
ax2.set(ylabel='Silhouette score', xlabel='Number of clusters', xticks=[2, 8, 14])

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_path, 'kmeans_inertia.jpg'), dpi=600)

# Clustering
psth_clusters = KMeans(n_clusters=CLUSTER_N, random_state=22, n_init='auto').fit_predict(dim_red_psth)
psth_df['cluster'] = psth_clusters

# %% Plot

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6.2, 3.5), dpi=dpi)
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

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='modulation', ax=ax4, legend=None,
                palette='coolwarm')
ax4.axis('off')
ax4.set(title='Modulation')

sns.scatterplot(data=psth_df, x='tsne_1', y='tsne_2', hue='high_level_region', ax=ax5,
                hue_order=np.unique(psth_df['high_level_region']),
                palette=[colors.get(key) for key in np.unique(psth_df['high_level_region'])])
ax5.legend(frameon=False, prop={'size': 5}, bbox_to_anchor=(1, 1))
ax5.set(title='Region')
#ax2.legend(frameon=False, prop={'size': 5})
ax5.axis('off')

ax6.set_axis_off()

plt.tight_layout()
plt.savefig(join(fig_path, 'tsne_embedding.jpg'), dpi=600)


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

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
#ax1.plot([0, 0], [-6, 6], ls='--', color='grey', lw=0.75)
for i in np.unique(psth_clusters):
    these_psth = all_psth[psth_clusters == i]
    ax1.fill_between(time_ax, np.mean(these_psth, axis=0) - (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    np.mean(these_psth, axis=0) + (np.std(these_psth, axis=0) / np.sqrt(these_psth.shape[0])),
                    alpha=0.25)
    ax1.plot(time_ax, np.mean(these_psth, axis=0), label='NS')
ax1.set(ylabel='Baseline subtracted spks/s', xlabel='Time (s)', xticks=[-1, 0, 1, 2, 3, 4])
plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'cluster_psth.jpg'), dpi=600)

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.pie(np.bincount(psth_df.loc[psth_df['type'] == 'NS', 'cluster']), colors=sns.color_palette())
ax1.set(title='NS')

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
