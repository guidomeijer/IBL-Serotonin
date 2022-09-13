#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:29:22 2022
By: Guido Meijer
"""

from os import mkdir
from os.path import join, isdir, split
from glob import glob
import shutil

DATA_DIR = '/media/guido/Data2/Facemap'
TARGET_DIR = '/media/guido/Data2/PassiveVideos/mainenlab/passive'

video_paths = glob(join(DATA_DIR, '*.mp4'))

for i, path in enumerate(video_paths):
    subject = split(path)[-1][:9]
    session = split(path)[-1][10:20]
    if not isdir(join(TARGET_DIR, subject)):
        mkdir(join(TARGET_DIR, subject))
    if not isdir(join(TARGET_DIR, subject, session)):
        mkdir(join(TARGET_DIR, subject, session))
    print(f'Copying {split(path)[-1]} ({i+1} of {len(path)})')
    shutil.copyfile(path, join(TARGET_DIR, subject, session, 'left_short.mp4'))


