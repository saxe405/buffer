# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:28:18 2019

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
from common_analysis_fns import Action_table_builder
from channel import get_CQ_matrix
import statistics

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def save_fig_cq(lab_vals, x_chart_vals, filename, ylabel, title, ylimits):
    labels = [str(int(100*x)) + '%' for x in pi1_range]   
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1*width/8, x_chart_vals[0], width/4, label='predictive-buffering', color = 'tab:blue')
    rects2 = ax.bar(x + 1*width/8, x_chart_vals[1], width/4, label='short-term-buffering', color = 'tab:orange')
    rects3 = ax.bar(x + 3*width/8, x_chart_vals[2], width/4, label='no-buffering', color = 'tab:green')
    rects4 = ax.bar(x - 3*width/8, x_chart_vals[3], width/4, label='benchmark', color = 'tab:purple')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel("% time network in bad state")
    ax.set_ylim(ylimits)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(False)    
    fig.tight_layout()    
    fig.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".pdf")
                    
reward_penalty_coeff = 160
l_max_range = [0,5]
pi1_range= [ round(0.05*x,2) for x in range(2,8)] 
user_gp = 1
video_gp = 1
mtrust = 0.8
npl = 'W3_all'
h_range = [0,1]
Action_tables_cq = {}

cachin_states = {}
streamin_states = {}
Cost_vec= {}
FOV_quality = {}
state_vec = {}
Cost_vec_stream= {}
FOV_quality_stream = {}
state_vec_stream = {}


print("Building all the action tables")
for l_m in l_max_range:
    for pi1 in pi1_range:
        for h in h_range:
            key = (pi1,l_m,h)
            Action_tables_cq[key] = Action_table_builder(Utility_book_name(reward_penalty_coeff,l_m,user_gp,video_gp,pi1,mtrust,h,npl))
            cachin_states[key]  = starting_State_f(T = max_tiles(video_group = video_gp))                
            Cost_vec[key]       = [0]
            FOV_quality[key]    = []
            state_vec[key]      = []        

for pi1 in pi1_range:
    streamin_states[pi1]      = s_state(max_tiles(video_group = video_gp), Q, starting_State_f().CQ)
    Cost_vec_stream[pi1]      = []
    FOV_quality_stream[pi1]    = []
    state_vec_stream[pi1]      = []


print("Action tables ready")
t = 0

print("Starting with the initial phase")
next_channel_quality = {}
while t < 20000:
    next_states = {}
    next_states_stream = {}
    for l_m in l_max_range:
        for pi1 in pi1_range:
            for h in h_range:                
                key = (pi1,l_m,h)
                next_states[key] = cache_state_without_NP_and_CQ(Action_tables_cq[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = video_gp))

    for pi1 in pi1_range:    
        next_states_stream[pi1] = stream_state_without_NP_and_CQ(streamin_states[pi1], video_group = video_gp)
    
    head_movement = {1 : np.random.choice(list(head_movement_lookup_table[1].keys()), p=list(head_movement_lookup_table[1].values()))}
    
    if t%5 == 0:
        for pi1 in pi1_range:
            CQ_mat = get_CQ_matrix_pi(pi1)   
            next_channel_quality[pi1] = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[(pi1,0,1)].CQ][0,x] for x in range(CQ_mat.shape[0])] )
    
    for l_m in l_max_range:       
        for pi1 in pi1_range:
            for h in h_range:                
                key = (pi1,l_m,h)
                if h == 0:
                    cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality[pi1])    
                else:                
                    cachin_states[key] = cache_state_NP( next_states[key], head_movement[1], next_channel_quality[pi1])    
    
    for pi1 in pi1_range:
        streamin_states[pi1] = stream_state_NP( next_states_stream[pi1], head_movement[1],next_channel_quality[pi1])
    t+=1
            
print("starting with the simulation phase")
t = 0
num_simulations = 100000

while t < num_simulations:    
    next_states = {}
    next_states_stream = {}
    for l_m in l_max_range:
        for pi1 in pi1_range:
            for h in h_range:                
                key = (pi1,l_m,h)
                next_states[key] = cache_state_without_NP_and_CQ(Action_tables_cq[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = video_gp))
    
    for pi1 in pi1_range:    
        next_states_stream[pi1] = stream_state_without_NP_and_CQ(streamin_states[pi1], video_group = video_gp)
        
    head_movement = {1 : np.random.choice(list(head_movement_lookup_table[1].keys()), p=list(head_movement_lookup_table[1].values()))}   
    
    if t%5 == 0:
        for pi1 in pi1_range:
            CQ_mat = get_CQ_matrix_pi(pi1)
            next_channel_quality[pi1] = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[(pi1,0,1)].CQ][0,x] for x in range(CQ_mat.shape[0])] )
    
    for l_m in l_max_range:       
        for pi1 in pi1_range:
            for h in h_range:               
                key = (pi1,l_m,h)
                Cost_vec[key].append( Networking_Cost_CQ( Action_tables_cq[key][cachin_states[key]], cachin_states[key].CQ))                        
                if h == 0:
                    cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality[pi1])    
                else:                        
                    cachin_states[key] = cache_state_NP( next_states[key], head_movement[1], next_channel_quality[pi1])    
                state_vec[key].append(cachin_states[key])
                if cachin_states[key].R == 1:
                    first_seg = cachin_states[key].n1
                    next_action = Action_tables_cq[key][cachin_states[key]]
                    if next_action[0] == 1:
                        first_seg +=next_action[1]
                    FOV_quality[key].append(FOV_quality_fn(first_seg, video_group = video_gp))
            
    for pi1 in pi1_range:
        Cost_vec_stream[pi1].append( Networking_Cost_stream(streamin_states[pi1], next_states_stream[pi1]))  
        streamin_states[pi1] = stream_state_NP( next_states_stream[pi1], head_movement[1],next_channel_quality[pi1])         
        if streamin_states[pi1].R == 1:
            first_seg = streamin_states[pi1].n1
            first_seg +=stream_action(streamin_states[pi1], video_group = video_gp)
            FOV_quality_stream[pi1].append(FOV_quality_stream_fn(first_seg, video_group = video_gp))
        state_vec_stream[pi1].append(streamin_states[pi1])
    t+=1

save_fig_cq( pi1_range,
             [[ statistics.mean(FOV_quality[(pi1,5,1)]) for pi1 in pi1_range],
             [ statistics.mean(FOV_quality[(pi1,0,1)]) for pi1 in pi1_range],
             [ statistics.mean(FOV_quality_stream[pi1]) for pi1 in pi1_range],
             [ statistics.mean(FOV_quality[(pi1,5,0)]) for pi1 in pi1_range]
             ],
             "QoEbarchart",
             'FoV quality',
             'Change in QoE with increasing unreliable connection',
             (0.4,0.9)
        )

save_fig_cq( pi1_range,
             [[ statistics.mean(Cost_vec[(pi1,5,1)]) for pi1 in pi1_range],
             [ statistics.mean(Cost_vec[(pi1,0,1)]) for pi1 in pi1_range],
             [ statistics.mean(Cost_vec_stream[pi1]) for pi1 in pi1_range],
             [ statistics.mean(Cost_vec[(pi1,5,0)]) for pi1 in pi1_range]
             ],
             "Costbarchart",
             'Resource usage',
             'Change in resource usage with increasing unreliable connection',
             (4,14)
        )
