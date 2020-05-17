# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:29:49 2020

@author: sapoorv
"""

import matplotlib.pyplot as plt
import re
import xlrd
__all__ = ["Identity", "Neighbours", "Probability_matrix"]
from Definitions import *
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from transition_rules import cache_state_without_NP_and_CQ
from transition_rules import cache_state_NP
import seaborn as sns
import time
import numpy as np
from shutil import copyfile
from reward_rules import Networking_Cost_CQ
import os
from streaming import *
from head_movements import *
from head_movements import compute_head_movement_table
from common_analysis_fns import Action_table_builder, read_traces
import statistics

def save_figure(lab_vals, x_chart_vals, filename, ylabel, title, ylimits):
       
    x = np.arange(len(lab_vals))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1*width/8, x_chart_vals[0], width/4, label='predictive-buffering', color = 'tab:blue')
    rects2 = ax.bar(x + 1*width/8, x_chart_vals[1], width/4, label='short-term-buffering', color = 'tab:orange')
    rects3 = ax.bar(x + 3*width/8, x_chart_vals[2], width/4, label='no-buffering', color = 'tab:green')
    rects4 = ax.bar(x - 3*width/8, x_chart_vals[3], width/4, label='benchmark', color = 'tab:purple')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Head movement scenario")
    
    ax.set_title(title)
    ax.set_xticks(x)
    if ylimits[1] < 1.5:
        y = [0.1*x for x in range(11)]
    else:
        y = [x for x in range(math.ceil(ylimits[1]))]
    ax.set_yticks(y)
    ax.set_xticklabels(lab_vals)
    ax.set_ylim(ylimits)
    ax.legend(loc='upper center', ncol=2, fancybox=True, shadow=True)    
    ax.grid(False)
    
    fig.tight_layout()
    
    fig.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".pdf")
    
Action_tables = {}
cachin_states = {}
streamin_states = {}
Cost_vec= {}
Freeze_prob = {}
FOV_quality = {}
state_vec = {}

Cost_vec_stream= {}
FOV_quality_stream = {}
Freeze_prob_stream = {}
state_vec_stream = {}

pi0 = 0.2
mtrust = 0.8
p = 120
ug = 1
vg = 1
l_max_range = [0,5]
h_range = [0,1]
scenarios = ['W3_all', 'W3_V1', 'W3_V2', 'W3_P17', 'W3_P21','W3_P25']
ta = True
print("Building all the action tables")

for npl in scenarios:
    for l_m in l_max_range:            
        for h in h_range:
            key = (l_m,h,npl) 
            '''
            The key is a tuple of 
                p - penalty parameter, 
                l_m : the length of longterm buffer
                ug : user group
                vg : video group
                h : 0 if no head movements, 1 if there are head_movements from the user (implemented to highlight the impact of head movement on performance)
            '''
            Action_tables[key]   = Action_table_builder(Utility_book_name(p,l_m,ug,vg,0.2,0.8,h,npl))
            cachin_states[key]  = starting_State_f(T = max_tiles(video_group = vg))                
            Cost_vec[key]       = [0]
            FOV_quality[key]    = []
            Freeze_prob[key]    = []
            state_vec[key]      = []            

for npl in scenarios:
    key = npl
    streamin_states[key]       = s_state(max_tiles(video_group = vg), Q, starting_State_f().CQ)
    Cost_vec_stream[key]       = [0]
    FOV_quality_stream[key]    = []
    Freeze_prob_stream[key]    = []
    state_vec_stream[key]      = []
        
hm_table = {}
for npl in scenarios:
    hm_table[npl] = compute_head_movement_table(npl = npl)
print("Action tables ready")
t = 0

print("Starting with the initial phase")

while t < 10000:
    channel_quality = cachin_states[ (l_max_range[0],1, scenarios[0]) ].CQ
    next_states = {}    
    next_states_stream = {}
    
    for npl in scenarios:
        for l_m in l_max_range:   
            for h in h_range:
                key = (l_m, h, npl)
                next_states[key] = cache_state_without_NP_and_CQ(Action_tables[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = vg))                    
    for npl in scenarios:
        key = npl
        next_states_stream[key] = stream_state_without_NP_and_CQ(streamin_states[key], video_group = vg)
    
    head_movement = {}
    for npl in scenarios:
        head_movement[npl] = np.random.choice(list(hm_table[npl][1].keys()), p=list(hm_table[npl][1].values()))
    
    if t%5 == 0:
        next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )
    
    for npl in scenarios:
        for l_m in l_max_range:
            for h in h_range:
                key = (l_m,h,npl)
                Cost_vec[key].append( Networking_Cost_CQ( Action_tables[key][cachin_states[key]], channel_quality))
                if h == 0:
                    cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality)
                else:
                    cachin_states[key] = cache_state_NP( next_states[key], head_movement[npl], next_channel_quality)                           
    for npl in scenarios:
        key = npl
        Cost_vec_stream[key].append( Networking_Cost_stream(streamin_states[key], next_states_stream[key]))
        streamin_states[key] = stream_state_NP( next_states_stream[key], head_movement[npl],next_channel_quality)
    t+=1

num_simulations = 50000
if ta:
    tn = 2
    print("reading 5G traces")
    Channel_path = read_traces(tn)    
    num_simulations = len(Channel_path)*5
print("starting with the simulation phase")
t = 0

while t < num_simulations:
    
    channel_quality = cachin_states[ (l_max_range[0],1, scenarios[0]) ].CQ
    next_states = {}
    for npl in scenarios:
        for l_m in l_max_range:   
            for h in h_range:
                key = (l_m, h, npl)
                next_states[key] = cache_state_without_NP_and_CQ(Action_tables[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = vg))   
    
    for npl in scenarios:
        key = npl
        next_states_stream[key] = stream_state_without_NP_and_CQ(streamin_states[key], video_group = vg)
    
    head_movement = {}
    for npl in scenarios:
        head_movement[npl] = np.random.choice(list(hm_table[npl][1].keys()), p=list(hm_table[npl][1].values()))
       
    if t%5 == 0:
        if ta:
            next_channel_quality = Channel_path.pop(0)
        else:
            next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

    for npl in scenarios:
        for l_m in l_max_range:
            for h in h_range:
                key = (l_m,h,npl)
                Cost_vec[key].append( Networking_Cost_CQ( Action_tables[key][cachin_states[key]], channel_quality))                
                Freeze_prob[key].append(Prob_screen_freeze(cachin_states[key], video_group = vg))
                if h == 0:
                    cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality)
                else:
                    cachin_states[key] = cache_state_NP( next_states[key], head_movement[npl], next_channel_quality)
                if cachin_states[key].R == 1:
                    first_seg = cachin_states[key].n1  
                    next_action = Action_tables[key][cachin_states[key]]
                    if next_action[0] == 1:
                        first_seg +=next_action[1]
                    FOV_quality[key].append(FOV_quality_fn(first_seg, video_group = vg))
                state_vec[key].append(cachin_states[key])
    for npl in scenarios:        
        key = npl
        Cost_vec_stream[key].append( Networking_Cost_stream(streamin_states[key], next_states_stream[key]))
        streamin_states[key] = stream_state_NP( next_states_stream[key], head_movement[npl],next_channel_quality)
        if streamin_states[key].R == 1:
            first_seg = streamin_states[key].n1
            first_seg +=stream_action(streamin_states[key], video_group = vg)
            FOV_quality_stream[key].append(FOV_quality_stream_fn(first_seg, video_group = vg))
        Freeze_prob_stream[key].append(Prob_screen_freeze_stream(streamin_states[key], video_group = vg))
        state_vec_stream[key].append(streamin_states[key])
    t+=1

save_figure(
      scenarios,
      [       
       [ statistics.mean(FOV_quality[(5,1,npl)]) for npl in scenarios],
       [ statistics.mean(FOV_quality[(0,1,npl)]) for npl in scenarios],
       [ statistics.mean(FOV_quality_stream[npl]) for npl in scenarios],
       [ statistics.mean(FOV_quality[(5,0,npl)]) for npl in scenarios]              
      ], 
      'FOV_quality_scenarios' + ('_ta' + str(tn) if ta else ''),
      'FOV quality',
       'FOV quality for different head movement probabilities',
       (0.5,1.1) if ta else (0.5,0.87)
       )
save_figure(
      scenarios,
      [       
       [ statistics.mean(Cost_vec[(5,1,npl)]) for npl in scenarios],
       [ statistics.mean(Cost_vec[(0,1,npl)]) for npl in scenarios],
       [ statistics.mean(Cost_vec_stream[npl]) for npl in scenarios],
       [ statistics.mean(Cost_vec[(5,0,npl)]) for npl in scenarios]              
      ], 
      'Cost_scenarios'+ ('_ta' + str(tn) if ta else ''),
      'Resource Usage',
       'Resource Usage for different head movement probabilities',
       (0.5,12) if ta else (5,12)
       )