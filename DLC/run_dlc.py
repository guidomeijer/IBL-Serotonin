#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:00:29 2021

@author: guido
"""

from iblvideo import run_session
from one.api import ONE
one = ONE()

#sessions = one.search(task_protocol='_iblrig_NPH_tasks_')
#sessions = one.search(project='serotonin_inference', task_protocol='biased')
sessions = one.search(project='serotonin_inference', task_protocol='_iblrig_tasks_opto_ephysChoiceWorld')

for i, eid in enumerate(sessions):
    print(f'\n\nProcessing session {eid} [{i + 1} of {len(sessions)}]\n\n')
    status = run_session(eid, machine='guido', cams=['left'], one=one, frames=10000, clobber=True,
                         overwrite=False)

