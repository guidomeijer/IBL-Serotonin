#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
from dlc_functions import get_dlc_XYs, get_pupil_diameter
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from serotonin_functions import load_trials, paths, DATE_GOOD_OPTO
from one.api import ONE
one = ONE()

# Settings
_, fig_path, _ = paths()
fig_path = join(fig_path, 'Pupil')

subjects = pd.read_csv(join('..', 'subjects.csv'))
subjects = subjects[subjects['subject'] == 'ZFM-02600'].reset_index(drop=True)
results_df = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids, details = one.search(subject=nickname, project='serotonin_inference',
                               task_protocol='biased', details=True)

    # Loop over sessions
    pupil_size = pd.DataFrame()
    for j, eid in enumerate(eids):
        print(f'Processing session {j+1} of {len(eids)}')
        rig = one.get_details(eid, full=True)['location']
        if 'ephys' not in rig:
            continue

        # Load in camera timestamps and DLC output
        try:
            video_times, XYs = get_dlc_XYs(eid, one=one)
        except:
            print('Could not load video and/or DLC data')
            continue

        # Get pupil diameter
        diameter = get_pupil_diameter(XYs)

        # Add to dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'subject': nickname, 'date': details[j]['date'],
            'opto': 'opto' in details[j]['task_protocol'],
            'diameter_mean': np.mean(diameter), 'diameter_std': np.std(diameter)}))



