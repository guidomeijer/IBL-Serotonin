#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:04:11 2022
By: Guido Meijer
"""

from os.path import join
from skimage import io

PATH = '/home/guido/Histology'


subject = 'ZFM-01802'
im = io.imread(join(PATH, subject, 'STD_ds_ZFM-01802_GR.tif'))