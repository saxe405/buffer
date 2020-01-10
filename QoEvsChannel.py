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

def save_fig_cq(lab_vals, x_chart_vals_1, x_chart_vals_2,x_chart_vals_3, filename, ylabel, title, ylimits):
    labels = [str(int(100*x)) + '%' for x in pi1_range]   
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2*width/3, x_chart_vals_1, 2*width/3, label='predictive-buffering', color = 'tab:blue')
    rects2 = ax.bar(x            , x_chart_vals_2, 2*width/3, label='short-term-buffering', color = 'tab:orange')
    rects3 = ax.bar(x + 2*width/3, x_chart_vals_3, 2*width/3, label='no-buffering', color = 'tab:green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel("% time network in bad state")
    ax.set_ylim(ylimits)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    fig.tight_layout()
    
    fig.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".png")
                    
reward_penalty_coeff = 160
l_max_range = [0,5]
pi1_range= [ round(0.05*x,2) for x in range(2,8)] 
user_gp = 1
video_gp = 1
mtrust = 0.8
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
        Action_tables_cq[(pi1,l_m)]   = Action_table_builder(Utility_book_name(reward_penalty_coeff,l_m,user_gp,video_gp,pi1,mtrust))

for l_m in l_max_range:
    for pi1 in pi1_range:
        cachin_states[(pi1,l_m)]  = starting_State_f(T = max_tiles(video_group = video_gp))                
        Cost_vec[(pi1,l_m)]       = [0]
        FOV_quality[(pi1,l_m)]    = []
        state_vec[(pi1,l_m)]      = []        

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
            next_states[(pi1,l_m)] = cache_state_without_NP_and_CQ(Action_tables_cq[(pi1,l_m)][cachin_states[(pi1,l_m)]], cachin_states[(pi1,l_m)], T = max_tiles(video_group = video_gp))

    for pi1 in pi1_range:    
        next_states_stream[pi1] = stream_state_without_NP_and_CQ(streamin_states[pi1], video_group = video_gp)

    head_movement = {1 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))}
    #head_movement[0] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[0].values())))], p=list(head_movement_lookup_table[0].values()))
    #head_movement[2] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[2].values())))], p=list(head_movement_lookup_table[2].values()))
    
    if t%5 == 0: #channel state is twice the length now
        for pi1 in pi1_range:
            #print([pi1, 0.2*(1-pi1), 0.8*(1-pi1)])
            #CQ_mat = get_CQ_matrix([pi1, 0.2*(1-pi1), 0.8*(1-pi1)])        
            CQ_mat = get_CQ_matrix_pi(pi1)   
            next_channel_quality[pi1] = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[(pi1,0)].CQ][0,x] for x in range(CQ_mat.shape[0])] )
    
    for l_m in l_max_range:       
        for pi1 in pi1_range:
            #Cost_vec[(pi1,l_m)].append( Networking_Cost_CQ( Action_tables_cq[(pi1,l_m)][cachin_states[(pi1,l_m)]], cachin_states[(pi1,l_m)].CQ))
            cachin_states[(pi1,l_m)] = cache_state_NP( next_states[(pi1,l_m)], head_movement[1], next_channel_quality[pi1])    
    
    for pi1 in pi1_range:
        #Cost_vec_stream[pi1].append( Networking_Cost_stream(streamin_states[pi1], next_states_stream[pi1]))  
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
            next_states[(pi1,l_m)] = cache_state_without_NP_and_CQ(Action_tables_cq[(pi1,l_m)][cachin_states[(pi1,l_m)]], cachin_states[(pi1,l_m)], T = max_tiles(video_group = video_gp))
    
    for pi1 in pi1_range:    
        next_states_stream[pi1] = stream_state_without_NP_and_CQ(streamin_states[pi1], video_group = video_gp)
    
    head_movement = {1 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))}
    #head_movement[0] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[0].values())))], p=list(head_movement_lookup_table[0].values()))
    #head_movement[2] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[2].values())))], p=list(head_movement_lookup_table[2].values()))
    if t%5 == 0:
        for pi1 in pi1_range:
            #CQ_mat = get_CQ_matrix([pi1, 0.2*(1-pi1), 0.8*(1-pi1)])        
            CQ_mat = get_CQ_matrix_pi(pi1)
            next_channel_quality[pi1] = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[(pi1,0)].CQ][0,x] for x in range(CQ_mat.shape[0])] )
    
    for l_m in l_max_range:       
        for pi1 in pi1_range:
            Cost_vec[(pi1,l_m)].append( Networking_Cost_CQ( Action_tables_cq[(pi1,l_m)][cachin_states[(pi1,l_m)]], cachin_states[(pi1,l_m)].CQ))                        
            cachin_states[(pi1,l_m)] = cache_state_NP( next_states[(pi1,l_m)], head_movement[1], next_channel_quality[pi1])    
            state_vec[(pi1,l_m)].append(cachin_states[(pi1,l_m)])
            if cachin_states[(pi1,l_m)].R == 1:
                first_seg = cachin_states[(pi1,l_m)].n1
                next_action = Action_tables_cq[(pi1,l_m)][cachin_states[(pi1,l_m)]]
                if next_action[0] == 1:
                    first_seg +=next_action[1]
                FOV_quality[(pi1,l_m)].append(FOV_quality_fn(first_seg, video_group = video_gp))
            
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
             [ statistics.mean(FOV_quality[(pi1,5)]) for pi1 in pi1_range],
             [ statistics.mean(FOV_quality[(pi1,0)]) for pi1 in pi1_range],
             [ statistics.mean(FOV_quality_stream[pi1]) for pi1 in pi1_range],
             "QoEbarchart",
             'FoV quality',
             'Change in QoE with increasing unreliable connection',
             (0.4,0.9)
        )

save_fig_cq( pi1_range,
             [ statistics.mean(Cost_vec[(pi1,5)]) for pi1 in pi1_range],
             [ statistics.mean(Cost_vec[(pi1,0)]) for pi1 in pi1_range],
             [ statistics.mean(Cost_vec_stream[pi1]) for pi1 in pi1_range],
             "Costbarchart",
             'Resource usage',
             'Change in costs with increasing unreliable connection',
             (0,4.5)
        )

#save_fig_cq( pi1_range,
#             [ statistics.variance(FOV_quality[(pi1,5)]) for pi1 in pi1_range],
#             [ statistics.variance(FOV_quality[(pi1,0)]) for pi1 in pi1_range],
#             [ statistics.variance(FOV_quality_stream[pi1]) for pi1 in pi1_range],
#             "QoEbarchart_variance",
#             'FoV quality',
#             'Variance in QOE with increasing unreliable connection',
#             (0.02,1)
#        )