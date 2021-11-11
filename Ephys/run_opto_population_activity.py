#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
from os import mkdir
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from serotonin_functions import paths, remap, query_ephys_sessions, load_opto_times
from one.api import ONE
from ibllib.atlas import AllenAtlas
lda = LinearDiscriminantAnalysis()
one = ONE()
ba = AllenAtlas()

# Settings
MIN_NEURONS = 1  # per region
PLOT = False
T_BEFORE = 1
T_AFTER = 2
BASELINE = 0.5
BIN_SIZE = 0.025
_, fig_path, save_path, repo_path = paths(return_repo_path=True)
fig_path = join(fig_path, 'Ephys', 'Population', 'LightMod')
save_path = join(save_path)

# Query sessions
eids, _ = query_ephys_sessions(one=one)

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

artifact_neurons = pd.read_csv(join(repo_path, 'artifact_neurons.csv'))
pop_act_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    print(f'Starting {subject}, {date}')

    # Load in laser pulse times
    try:
        opto_train_times = load_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, dataset_types=['spikes.amps', 'spikes.depths'], one=one, brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):

        # Filter neurons that pass QC
        if 'metrics' in clusters[probe].keys():
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        else:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]

        # Select spikes of passive period
        start_passive = opto_train_times[0] - 360
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > start_passive]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > start_passive]

        # Exclude artifact neurons
        clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
            (artifact_neurons['eid'] == eid) & (artifact_neurons['probe'] == probe), 'neuron_id'].values])
        if clusters_pass.shape[0] == 0:
            continue

        # Get regions from Beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'])
        clusters_regions = clusters[probe]['acronym'][clusters_pass]

        # Loop over regions
        for r, region in enumerate(np.unique(clusters_regions)):
            if region == 'root':
                continue

            # Select spikes and clusters in this brain region
            clusters_in_region = clusters_pass[clusters_regions == region]
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
            if len(clusters_in_region) < MIN_NEURONS:
                continue

            # Get spike vector
            peth, _ = calculate_peths(spks_region, np.ones(spks_region.shape), [1], opto_train_times,
                                       T_BEFORE, T_AFTER, BIN_SIZE)
            pop_act = peth['means'][0]
            time = peth['tscale']
            pop_act_baseline = pop_act - np.median(pop_act[(time > -BASELINE) & (time < 0)])
            pop_act_df = pop_act_df.append(pd.DataFrame(data={
                'subject': subject, 'date': date, 'probe': probe, 'eid': eid,
                'pop_act': pop_act, 'region': region, 'time': time, 'pop_act_baseline': pop_act_baseline}))

            # Plot
            colors, dpi = figure_style()
            f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
            peri_event_time_histogram(spks_region, np.ones(spks_region.shape), opto_train_times, 1,
                                      t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
                                      error_bars='sem', pethline_kwargs={'color': 'black', 'lw': 1},
                                      errbar_kwargs={'color': 'black', 'alpha': 0.3},
                                      eventline_kwargs={'lw': 0})
            ax.set(ylim=[ax.get_ylim()[0], ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2])
            ax.plot([0, 1], [ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05,
                             ax.get_ylim()[1] - ax.get_ylim()[1] * 0.05], lw=2, color='royalblue')
            ax.set(ylabel='Population activity (spks/s)', xlabel='Time (s)', title=region,
                    yticks=np.linspace(0, np.round(ax.get_ylim()[1]), 3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.tight_layout()
            plt.savefig(join(fig_path, f'{region}_{subject}_{date}_{probe}.pdf'))
            plt.close(f)

pop_act_df.to_csv(join(save_path, 'pop_act_opto_per_region.csv'), index=False)

# %% Plot all regions
pop_act_df = pop_act_df.reset_index(drop=True)

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)
sns.lineplot(x='time', y='pop_act_baseline', data=pop_act_df, ci=68, hue='region', estimator=np.nanmean)
ax1.set(xlabel='Time (s)', ylabel='Baseline subtracted pop. activity (spks/s)')


