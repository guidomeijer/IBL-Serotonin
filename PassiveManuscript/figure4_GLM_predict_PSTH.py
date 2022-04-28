# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:00:14 2022

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import paths, remap,  load_passive_opto_times
from glm_functions import GLMPredictor
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Amygdala enhanced neuron
SUBJECT = 'ZFM-01802'
DATE = '2021-03-11'
PROBE = 'probe00'
NEURON = 226

# Settings
T_BEFORE = 1  # for plotting
T_AFTER = 2
ZETA_BEFORE = 0  # baseline period to include for zeta test
PRE_TIME = [1, 0]  # for modulation index
POST_TIME = [0, 1]
BIN_SIZE = 0.05
SMOOTHING = 0.025
fig_path, save_path = paths(dropbox=True)
fig_path = join(fig_path, 'PaperPassive', 'figure4')

# Get session details
ins = one.alyx.rest('insertions', 'list', date=DATE, subject=SUBJECT, name=PROBE)
pid = ins[0]['id']
eid = ins[0]['session']

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in spikes
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Get region
region = remap(clusters.acronym[NEURON])[0]

GLMPredictor()

