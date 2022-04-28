#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import convolve, gaussian
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
import matplotlib.pyplot as plt
import seaborn as sns
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, figure_style, calculate_peths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
REGIONS = ['M2', 'mPFC', 'ORB', 'Amyg', 'Pir']
VAR_CUTOFF = 1
NEURON_QC = True  # whether to use neuron qc to exclude bad units
MIN_NEURONS = 10  # minimum neurons per region
N_PERMUT = 100  # number of times to get spontaneous population correlation for permutation testing
WIN_SIZE = 0.2  # window size in seconds
PRE_TIME = 1  # time before stim onset in s
POST_TIME = 2  # time after stim onset in s
SMOOTHING = 0.1  # smoothing of psth
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use
TEST_FRAC = 0.5  # fraction to use for testing in cross-validation
N_BOOTSTRAP = 50  # amount of times to bootstrap cross-validation
PLOT_IND = True  # plot individual region pairs

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA')

# Initialize some things
np.random.seed(42)  # fix random seed for reproducibility
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
lio = LeaveOneOut()
if SMOOTHING > 0:
    w = n_time_bins - 1 if n_time_bins % 2 == 0 else n_time_bins
    window = gaussian(w, std=SMOOTHING / WIN_SIZE)
    window /= np.sum(window)


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


# Query sessions with frontal and amygdala
rec = query_ephys_sessions(acronym=['MOs', 'BLA', 'MEA', 'CEA', 'ILA', 'PL', 'ACA'], one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

pca_df = pd.DataFrame()
for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Starting {subject}, {date}')
    
    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')
    
    # Load in spikes
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except:
        continue

    # Filter neurons that pass QC and artifact neurons
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                              cluster_ids=np.arange(clusters.channels.size))
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    clusters['region'] = remap(clusters['acronym'], combine=True, abbreviate=True)
    
    for j, region in enumerate(REGIONS):
        
        # Exclude neurons with low firing rates
        clusters_in_region = np.where(clusters['region'] == region)[0]
        fr = np.empty(clusters_in_region.shape[0])
        for nn, neuron_id in enumerate(clusters_in_region):
            fr[nn] = np.sum(spikes.clusters == neuron_id) / spikes.clusters[-1]
        clusters_in_region = clusters_in_region[fr >= MIN_FR]
        
        # Get spikes and clusters
        spks_region = spikes.times[np.isin(spikes.clusters, clusters_in_region) & np.isin(spikes.clusters, clusters_pass)]
        clus_region = spikes.clusters[np.isin(spikes.clusters, clusters_in_region) & np.isin(spikes.clusters, clusters_pass)]
        if np.unique(clus_region).shape[0] < MIN_NEURONS:
            continue        
        
        # Initialize PCA   
        n_components = np.min([np.unique(clus_region).shape[0], opto_train_times.shape[0]])
        if n_components > 50:
            n_components = 50
        pca = PCA(n_components=n_components)
   
        # Get binned spikes for SPONTANEOUS activity and subtract mean
        pca_spont = np.empty([N_PERMUT, n_time_bins])
        for jj in range(N_PERMUT):

            # Get random times for spontaneous activity
            spont_on_times = np.sort(np.random.uniform(
                opto_train_times[0] - (6 * 60), opto_train_times[0], size=opto_train_times.shape[0]))

            # Get PSTH and binned spikes for SPONTANEOUS activity
            psth_spont, binned_spks_spont = calculate_peths(
                spks_region, clus_region, np.unique(clus_region), spont_on_times, pre_time=PRE_TIME,
                post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)

            # Perform PCA
            for tb in range(binned_spks_spont.shape[2]):
                pca.fit(binned_spks_spont[:, :, tb])  
                pca_spont[jj, tb] = np.sum(pca.explained_variance_ratio_ > VAR_CUTOFF/100)
            
        # Get PSTH and binned spikes for OPTO activity
        psth_opto, binned_spks_opto = calculate_peths(
            spks_region, clus_region, np.unique(clus_region), opto_train_times, pre_time=PRE_TIME,
            post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)

        # Perform PCA
        pca_opto = np.empty(binned_spks_opto.shape[2])
        for tb in range(binned_spks_opto.shape[2]):
            pca.fit(binned_spks_opto[:, :, tb])
            pca_opto[tb] = np.sum(pca.explained_variance_ratio_ > VAR_CUTOFF/100)
        
        # Add to dataframe
        pca_df = pd.concat((pca_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'region': region,
            'pca_opto': pca_opto, 'pca_spont_mean': pca_spont.mean(axis=0),
            'pca_spont_05': np.quantile(pca_spont, 0.05, axis=0),
            'pca_spont_95': np.quantile(pca_spont, 0.95, axis=0),
            'time': psth_opto['tscale']})), ignore_index=True)
        
        # Plot this region 
        if PLOT_IND:
            colors, dpi = figure_style()
            f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
            #ax1.fill_between(psth_opto['tscale'], np.mean(r_spont, axis=0)-np.std(r_spont)/2,
            #                 np.mean(r_spont, axis=0)+np.std(r_spont)/2, color='grey', alpha=0.2)
            ax1.fill_between(psth_opto['tscale'], np.quantile(pca_spont, 0.05, axis=0),
                             np.quantile(pca_spont, 0.95, axis=0), color='grey', alpha=0.2)
            #ax1.plot(psth_opto['tscale'], np.mean(r_spont, axis=0), color='grey', lw=1)
            ax1.plot(psth_opto['tscale'], pca_opto, lw=2)
            ax1.plot([0, 0], ax1.get_ylim(), color='k', ls='--')
            ax1.set(xlabel='Time (s)', ylabel='Dimensionality of\npopulation activity',
                    title=f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(fig_path, 'Dimensionality', f'{region}_{subject}_{date}'), dpi=300)
            plt.close(f)
        
        # Save results
        pca_df.to_csv(join(save_path, 'pca_dimensionality.csv'))

