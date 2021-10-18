2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:47:37 2021
By: Guido Meijer
"""

from serotonin_functions import DATE_GOOD_OPTO
from os.path import join, isfile
import pandas as pd
from DLC_labeled_video import Viewer
from one.api import ONE
one = ONE()

video_save_path = '/home/guido/Data/5HT/DLC/'
trials = [10, 15]
subjects = pd.read_csv(join('..', 'subjects.csv'))

for i, nickname in enumerate(subjects['subject']):
    print(f'Processing {nickname}..')

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                      date_range=[DATE_GOOD_OPTO, '2025-01-01'])

    for j, eid in enumerate(eids):
        details = one.get_details(eid)
        date = details['date']

        if not isfile(join(video_save_path, f'{nickname}_trials_{trials[0]}_{trials[1]}_left.mp4')):
            Viewer(eid, 'left', [10, 15], video_save_path, eye_zoom=True)

