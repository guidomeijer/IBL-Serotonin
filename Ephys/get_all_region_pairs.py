#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:25:46 2022
By: Guido Meijer
"""

import numpy as np
from brainbox.io.one import SpikeSortingLoader
import pandas as pd
from serotonin_functions import query_ephys_sessions, remap, load_subjects
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

rec = query_ephys_sessions(one=one)
region_pair_df = pd.DataFrame()
for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    print(f'Starting {subject}, {date}')


    # Load in neural data of both probes of the recording
    acronyms = []
    for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

        try:
            sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)

        except Exception as err:
            print(err)
            continue

        acronyms.append(np.unique(clusters['acronym']))

    regions = np.unique(remap(np.concatenate(acronyms), combine=True))
    regions = regions[regions != 'root']
    region_pairs = []
    for k, region_1 in enumerate(regions):
        for m, region_2 in enumerate(regions[k:]):
            if region_1 != region_2:
                region_pairs.append(f'{region_1}-{region_2}')

    # Add to df
    region_pair_df = pd.concat((region_pair_df, pd.DataFrame(data={
        'eid': eid, 'subject': subject, 'date': date,
        'region_pair': region_pairs})))

# %% Results

# Add expression
subjects = load_subjects()
for i, nickname in enumerate(np.unique(subjects['subject'])):
    region_pair_df.loc[region_pair_df['subject'] == nickname, 'sert-cre'] = subjects.loc[subjects['subject'] == nickname, 'sert-cre'].values[0]

# Select sert mice and regions of interest
region_pair_df = region_pair_df[region_pair_df['sert-cre'] == 1]


grouped_df = region_pair_df.groupby('region_pair').size()
