# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:30:43 2022

@author: Guido
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from glob import glob
from brainbox.io.spikeglx import spikeglx
from serotonin_functions import paths, query_ephys_sessions, load_passive_opto_times
from one.api import ONE
one = ONE()

# Get path to figure save location
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Ephys', 'OptoPulses')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    
    # Get onset times using function
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    
    # Load in opto trace
    try:
        one.load_datasets(eid, datasets=[
            '_spikeglx_ephysData_g*_t0.nidq.cbin', '_spikeglx_ephysData_g*_t0.nidq.meta',
            '_spikeglx_ephysData_g*_t0.nidq.ch'], download_only=True)
    except:
        continue
    session_path = one.eid2path(eid)
    nidq_file = glob(str(session_path.joinpath('raw_ephys_data/_spikeglx_ephysData_g*_t0.nidq.cbin')))[0]
    sr = spikeglx.Reader(nidq_file)
    offset = int((sr.shape[0] / sr.fs - 720) * sr.fs)
    opto_trace = sr.read_sync_analog(slice(offset, sr.shape[0]))[:, 1]
    opto_times = np.arange(offset, sr.shape[0]) / sr.fs
    
    # Plot result
    f, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    ax1.plot(opto_times, opto_trace)
    ax1.plot(opto_train_times, np.zeros(len(opto_train_times)), 'xr', lw=2)
    ax1.set(title=f'{subject} {date}')
    
    plt.tight_layout()
    plt.savefig(join(fig_path, f'{subject}_{date}'))
    plt.close(f)
    