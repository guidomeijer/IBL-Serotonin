#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from brainbox.metrics.single_units import spike_sorting_metrics
import brainbox.io.one as bbone
from serotonin_functions import paths, query_ephys_sessions
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
NEURON_QC = True
_, _, save_path = paths()

# Query sessions
eids, _ = query_ephys_sessions(one=one)

waveforms_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

    print(f'Starting {subject}, {date}')

    # Load in spikes
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
           print(f'No brain regions found for {eid}')
           continue

        # Get data collection
        collections = one.list_collections(eid)
        if f'alf/{probe}/pykilosort' in collections:
            alf_path = one.eid2path(eid).joinpath('alf', probe, 'pykilosort')
            collection = f'alf/{probe}/pykilosort'
        else:
            alf_path = one.eid2path(eid).joinpath('alf', probe)
            collection = f'alf/{probe}'

        # Load in waveforms
        data = one.load_datasets(eid, datasets=['_phy_spikes_subset.waveforms', '_phy_spikes_subset.spikes',
                                                '_phy_spikes_subset.channels'],
                                 collections=[collection]*3)[0]
        waveforms, wf_spikes, wf_channels = data[0], data[1], data[2]
        waveforms = waveforms * 1000  # to uV

        # Filter neurons that pass QC
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        if len(spikes[probe].clusters) == 0:
            continue
        clusters_regions = clusters[probe]['acronym'][clusters_pass]

        # Loop over clusters
        for n, neuron_id in enumerate(clusters_pass):

            # Get mean waveform of channel with max amplitude
            n_waveforms = waveforms[spikes[probe].clusters[wf_spikes] == neuron_id].shape[0]
            if n_waveforms == 0:
                continue
            mean_wf_ch = np.mean(waveforms[spikes[probe].clusters[wf_spikes] == neuron_id], axis=0)
            mean_wf_ch = (mean_wf_ch
                          - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
            mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))]
            wf_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
            spike_amp = np.abs(np.min(mean_wf) - np.max(mean_wf))

            # Get peak-to-trough ratio
            pt_ratio = np.max(mean_wf) / np.abs(np.min(mean_wf))

            # Get peak minus through
            pt_subtract = np.max(mean_wf) - np.abs(np.min(mean_wf))

            # Get part of spike from trough to first peak after the trough
            peak_after_trough = np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)
            repolarization = mean_wf[np.argmin(mean_wf):np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)]

            # Get spike width in ms
            x_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
            peak_to_trough = ((np.argmax(mean_wf) - np.argmin(mean_wf)) / 30000) * 1000
            spike_width = ((peak_after_trough - np.argmin(mean_wf)) / 30000) * 1000

            # Get repolarization slope
            if spike_width <= 0.08:
                continue
            else:
                rp_slope, _, = np.polyfit(x_time[np.argmin(mean_wf):peak_after_trough],
                                          mean_wf[np.argmin(mean_wf):peak_after_trough], 1)

            # Get recovery slope
            rc_slope, _ = np.polyfit(x_time[peak_after_trough:], mean_wf[peak_after_trough:], 1)

            # Get firing rate
            neuron_fr = (np.sum(spikes[probe]['clusters'] == neuron_id)
                         / np.max(spikes[probe]['times']))

            # Add to dataframe
            waveforms_df = waveforms_df.append(pd.DataFrame(index=[waveforms_df.shape[0] + 1], data={
                'eid': eid, 'probe': probe, 'subject': subject, 'waveform': [mean_wf],
                'cluster_id': neuron_id, 'regions': clusters_regions[n], 'spike_amp': spike_amp,
                'pt_ratio': pt_ratio, 'rp_slope': rp_slope, 'pt_subtract': pt_subtract,
                'rc_slope': rc_slope, 'peak_to_trough': peak_to_trough, 'spike_width': spike_width,
                'firing_rate': neuron_fr, 'n_waveforms': n_waveforms}))

waveforms_df.to_pickle(join(save_path, 'waveform_metrics.p'))
