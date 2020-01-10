# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:23:35 2019

@author: sapoorv
"""

import pandas as pd
import os

files = []
path = "profilerLogs/"
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

for f in files:
    data = pd.read_csv(f)  
