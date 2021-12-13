#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
from brainbox.io.spikeglx import spikeglx
from ibllib.dsp.voltage import decompress_destripe_cbin
from serotonin_functions import paths, query_ephys_sessions
from shutil import copyfile
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = False
_, fig_path, save_path = paths()
save_path = join(save_path, 'LFP')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    ins = one.alyx.rest('insertions', 'list', session=eid)
    probes = [i['name'] for i in ins]

    for p, probe in enumerate(probes):

        # Check if artifact removal has already been done
        if (isfile(join(save_path, f'{subject}_{date}_{probe}_destriped_lfp.cbin'))) & ~OVERWRITE:
            print(f'{subject}, {date}, {probe} already destriped')
            continue

        try:
            # Get lfp data
            bin_path = one.load_dataset(eid, download_only=True, dataset=
                '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', collection=f'raw_ephys_data/{probe}')
            sr = spikeglx.Reader(bin_path)

            # Destriping settings
            butter_kwargs = {'N': 3, 'Wn': np.array([2, 125]) / sr.fs * 2, 'btype': 'bandpass'}
            k_kwargs = {'ntr_pad': 60, 'ntr_tap': 0, 'lagc': int(0.1 * sr.fs),
                        'butter_kwargs': {'N': 3, 'Wn': 0.001, 'btype': 'highpass'}}

            # Do destriping
            print(f'Destriping session {subject}, {date}, {probe}')
            decompress_destripe_cbin(bin_path,
                                     output_file=join(save_path, f'{subject}_{date}_{probe}_destriped_lfp.bin'),
                                     butter_kwargs=butter_kwargs, k_kwargs=k_kwargs, compute_rms=False)

            # Copy meta and channels file
            meta_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
               '_spikeglx_ephysData_g*_t0.imec*.lf.meta', '_spikeglx_ephysData_g*_t0.imec*.lf.ch'],
                collections=[f'raw_ephys_data/{probe}'] * 2)
            copyfile(meta_paths[0], join(save_path, f'{subject}_{date}_{probe}_destriped_lfp.meta'))
            copyfile(meta_paths[1], join(save_path, f'{subject}_{date}_{probe}_destriped_lfp.ch'))

        except:
            print(f'Error in session {subject}, {date}, {probe}')
            continue
