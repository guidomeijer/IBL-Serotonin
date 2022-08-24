#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:00:36 2022
By: Guido Meijer
"""

from os import listdir
from os.path import join, isfile, isdir
from shutil import copyfile
from glob import glob
from serotonin_functions import query_ephys_sessions
from one.api import ONE
one = ONE(mode='local')

# Settings
FROM = '/media/guido/Data2/FlatIron/mainenlab/Subjects'
TO = '/media/guido/IBLvideo2/5HT/VideoPassive'
OVERWRITE = False

# Get sessions
rec = query_ephys_sessions(one=one)

for i in rec.index.values:
    subject = rec.loc[i, 'subject']
    date = rec.loc[i, 'date']
    ses = listdir(join(FROM, subject, date))[0]
    if isdir(join(FROM, subject, date, ses, 'raw_video_data')):
        video_files = glob(join(FROM, subject, date, ses, 'raw_video_data', '*.mp4'))
    for j, video_file in enumerate(video_files):
        if video_file[-18:-14] == 'ight':
            cam = 'right'
        else:
            cam = video_file[-18:-14]
        if (isfile(join(TO, f'{subject}_{date}_{cam}.mp4')) == False) | OVERWRITE:
            print(f'Copying {subject}_{date}_{cam}.mp4')
            copyfile(video_file, join(TO, f'{subject}_{date}_{cam}.mp4'))


