# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:53:27 2019

@author: sapoorv
"""

import matplotlib.pyplot as plt

__all__ = ["Identity", "Neighbours", "Probability_matrix"]
from Definitions import *
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from transition_rules import cache_state_without_NP_and_CQ
from transition_rules import cache_state_NP
import numpy as np
from reward_rules import Networking_Cost_CQ
from streaming import *
from head_movements import *
from common_analysis_fns import Action_table_builder, read_traces, next_channel_quality_fn
from channel import get_CQ_matrix
import statistics

def save_fig_cq(x_chart_vals, filename, ylabel, title, ylim = 1, trace_analysis = False):
    labels = ['20%', '30%', '40%', '50%', '60%', '70%', '', '']    
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots()    
    rects1 = ax.bar(x[:-2]        , x_chart_vals[:-2], width, label='pred-caching')
    rects2 = ax.bar(x[-2:-1]      , x_chart_vals[-2:-1], width, label='short-caching')
    rects3 = ax.bar(x[-1:]        , x_chart_vals[-1:], width, label='no-caching')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,ylim)
    ax.set_xlabel("% usage of long term predictions")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    fig.tight_layout()
    if trace_analysis:
        filename += "TA"
    
    fig.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".png")
                    
reward_penalty_coeff = 160
l_m = 5
mt_range = [ round(0.1*x,1) for x in range(2,8)]
user_gp = 1
video_gp = 1
pi1 = 0.2
CQ_mat = get_CQ_matrix([pi1, 0.2*(1-pi1), 0.8*(1-pi1)])   
Action_tables_mt = {}

cachin_states = {}
streamin_states = {}
Cost_vec= {}
Freeze_prob = {}
state_vec = {}
Cost_vec_stream= {}
Freeze_prob_stream = {}
state_vec_stream = {}

print("Building all the action tables")
for mt in mt_range:
    Action_tables_mt[mt]   = Action_table_builder(Utility_book_name(reward_penalty_coeff,l_m,user_gp,video_gp,pi1,mt))
#for case where only short term predictions are used
Action_tables_mt[0]   = Action_table_builder(Utility_book_name(reward_penalty_coeff,0,user_gp,video_gp,pi1,0.2))

mt_range = mt_range + [0] #0 must remain at the end
for mt in mt_range:
    cachin_states[mt]  = starting_State_f(T = max_tiles(video_group = video_gp))                
    Cost_vec[mt]       = [0]
    Freeze_prob[mt]    = []
    state_vec[mt]      = []        

streamin_states      = s_state(max_tiles(video_group = video_gp), Q, starting_State_f().CQ)
Cost_vec_stream      = []
Freeze_prob_stream   = []
state_vec_stream     = []

print("Action tables ready")
t = 0

print("Starting with the initial phase")
next_states = {}
debugging = True
while t < 10000:
    t+=1    
    #next_states = {}    
    for mt in mt_range:
        next_states[mt] = cache_state_without_NP_and_CQ(Action_tables_mt[mt][cachin_states[mt]], cachin_states[mt], T = max_tiles(video_group = video_gp))        
    
    next_states_stream = stream_state_without_NP_and_CQ(streamin_states, video_group = video_gp)
    
    head_movement = {1 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))}
    next_channel_quality = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[mt].CQ][0,x] for x in range(CQ_mat.shape[0])] )
        
    for mt in mt_range:
        #Cost_vec[mt].append( Networking_Cost_CQ( Action_tables_mt[mt][cachin_states[mt]], cachin_states[mt].CQ))
        temp = cache_state_NP( next_states[mt], head_movement[1], next_channel_quality)
        if temp not in Action_tables_mt[mt] and temp.L >=5:
            Action_tables_mt[mt][temp] = (0,0)
            
        if temp not in Action_tables_mt[mt] and debugging:
            print(cachin_states[mt])
            print(Action_tables_mt[mt][cachin_states[mt]])
            print(temp)
            print(head_movement)
            
        cachin_states[mt] = temp
 
    #Cost_vec_stream.append( Networking_Cost_stream(streamin_states, next_states_stream))  
    streamin_states = stream_state_NP( next_states_stream, head_movement[1],next_channel_quality)
            
print("starting with the simulation phase")
t = 0

channel_vector = read_traces(5)
num_simulations = 500000
trace_analysis = False
if trace_analysis:
    num_simulations = min(num_simulations,len(channel_vector)*Q)

count = 0
while t < num_simulations:
    t+=1    
    next_states = {}
    next_states_stream = {}
    
    for mt in mt_range:
        next_states[mt] = cache_state_without_NP_and_CQ(Action_tables_mt[mt][cachin_states[mt]], cachin_states[mt], T = max_tiles(video_group = video_gp))
    
    next_states_stream = stream_state_without_NP_and_CQ(streamin_states, video_group = video_gp)
    
    head_movement = {1 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))}
    next_channel_quality = np.random.choice([x for x in range(CQ_mat.shape[0])], p=[CQ_mat[cachin_states[mt].CQ][0,x] for x in range(CQ_mat.shape[0])] )
    
    if trace_analysis:
        temp1 = next_channel_quality_fn(cachin_states[mt].CQ, channel_vector, cachin_states[mt].R)    
        next_channel_quality = temp1[0]
        channel_vector = temp1[1]
    
    for mt in mt_range:
        Cost_vec[mt].append( Networking_Cost_CQ( Action_tables_mt[mt][cachin_states[mt]], cachin_states[mt].CQ))            
        cachin_states[mt] = cache_state_NP( next_states[mt], head_movement[1], next_channel_quality)    
        if cachin_states[mt].R == 1:
            Freeze_prob[mt].append(Prob_screen_freeze(cachin_states[mt], video_group = video_gp))
        state_vec[mt].append(cachin_states[mt])
        if cachin_states[mt] not in Action_tables_mt[mt] and cachin_states[mt].L >=5:
            Action_tables_mt[mt][cachin_states[mt]] = (0,0)
 
    Cost_vec_stream.append( Networking_Cost_stream(streamin_states, next_states_stream))  
    streamin_states = stream_state_NP( next_states_stream, head_movement[1],next_channel_quality)
    if streamin_states.R == 1:
        Freeze_prob_stream.append(Prob_screen_freeze_stream(streamin_states, video_group = video_gp))
    state_vec_stream.append(streamin_states)


save_fig_cq( [ statistics.mean(Freeze_prob[mt]) for mt in mt_range] + [statistics.mean(Freeze_prob_stream)],
             "QoEbarchart_mt",
             'Freeze Probability',
             'Change in QoE with use of long term predictions',
             ylim = 1,
             trace_analysis = trace_analysis
        )

save_fig_cq( [ statistics.mean(Cost_vec[mt]) for mt in mt_range] + [statistics.mean(Cost_vec_stream)],            
             "Costbarchart_mt",
             'Networking cost',
             'Change in costs with use of long term predictions',
             ylim = 2,
             trace_analysis = trace_analysis,
        )
save_fig_cq( [ statistics.variance(Freeze_prob[mt]) for mt in mt_range] + [statistics.variance(Freeze_prob_stream)],
             "QoEbarchart_mt_variance",
             'Freeze Probability',
             'Variance of QoE with use of long term predictions',
             trace_analysis = trace_analysis
        )