#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:00:34 2022
By: Guido Meijer
"""

import os
import numpy as np
from serotonin_functions import query_ephys_sessions
from one.api import ONE
one = ONE()

rec = query_ephys_sessions()

for i, eid in enumerate(np.unique(rec['eid'])):
    one.load_dataset(eid, dataset='_iblrig_leftCamera.raw.mp4', download_only=True)
    #one.load_dataset(eid, dataset='_iblrig_rightCamera.raw.mp4', download_only=True)
    #one.load_dataset(eid, dataset='_iblrig_bodyCamera.raw.mp4', download_only=True)
