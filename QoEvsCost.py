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
import itertools

reward_penalty_coeffs = [ x for x in range(80,801,40) ]
reward_penalty_coeffs += [ x for x in range(900, 1201, 100)]
reward_penalty_coeffs = reward_penalty_coeffs[1:] #skip the first one as it has terrible performance
l_max_range = [0,5]
h_range = [0,1]

user_gps  = [1]
video_gps = [1]
npl = args.NP_log

def save_sub_figure(num, x_vecs, y_vecs, labels, x_lab, y_lab, file_name_suffix = ""):
    plt.style.use('seaborn')
    palette = plt.get_cmap('Set1')
    plt.figure(num)
    fig, ax = plt.subplots()
    ax.set_facecolor("w")
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for i in range(len(x_vecs)):
        ax.scatter( x_vecs[i], y_vecs[i] , color=colors[i], label = labels[i], alpha=0.9, marker =  ['d','*','.','+'][i])
    
    plt.xlabel(x_lab, fontsize=15)
    plt.ylabel(y_lab, fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim((0.6,0.92))
    plt.xlim((7.5,11))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)    
    plt.tight_layout()
    plt.grid(True)

    filename = y_lab + "vs"+ x_lab + file_name_suffix
    filename = filename.replace(" ", "_")
    plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".pdf")
    
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
print("Building all the action tables")
all_lists = [
        reward_penalty_coeffs,
        l_max_range,
        user_gps,
        video_gps,
        h_range
        ]
for element in itertools.product(*all_lists):
    p   = element[0]
    l_m = element[1]
    ug  = element[2]
    vg  = element[3]
    h   = element[4]

    key = (p,l_m,ug,vg,h) 
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

for ug in user_gps:
    for vg in video_gps:
        key = (ug,vg)
        streamin_states[key]       = s_state(max_tiles(video_group = vg), Q, starting_State_f().CQ)
        Cost_vec_stream[key]       = [0]
        FOV_quality_stream[key]    = []
        Freeze_prob_stream[key]    = []
        state_vec_stream[key]      = []
        

print("Action tables ready")
t = 0

print("Starting with the initial phase")

while t < 10000:
    channel_quality = cachin_states[ (reward_penalty_coeffs[0], l_max_range[0], user_gps[0], video_gps[0],1) ].CQ
    next_states = {}    
    next_states_stream = {}
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    for h in h_range:
                        key = (p,l_m,ug,vg,h)
                        next_states[key] = cache_state_without_NP_and_CQ(Action_tables[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = vg))                    
    for ug in user_gps:
        for vg in video_gps:
            next_states_stream[(ug,vg)] = stream_state_without_NP_and_CQ(streamin_states[(ug,vg)], video_group = vg)
    head_movement = {0 : np.random.choice(list(head_movement_lookup_table[0].keys()), p=list(head_movement_lookup_table[0].values()))}
    head_movement[1] = np.random.choice(list(head_movement_lookup_table[1].keys()), p=list(head_movement_lookup_table[1].values()))
    head_movement[2] = np.random.choice(list(head_movement_lookup_table[2].keys()), p=list(head_movement_lookup_table[2].values()))
    if t%5 == 0:
        next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )
    
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    for h in h_range:
                        key = (p,l_m,ug,vg,h)
                        Cost_vec[key].append( Networking_Cost_CQ( Action_tables[key][cachin_states[key]], channel_quality))

                        if h == 0:
                            cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality)
                        else:
                            cachin_states[key] = cache_state_NP( next_states[key], head_movement[ug], next_channel_quality)                           

    for ug in user_gps:
        for vg in video_gps:
                    Cost_vec_stream[(ug,vg)].append( Networking_Cost_stream(streamin_states[(ug,vg)], next_states_stream[(ug,vg)]))
                    streamin_states[(ug,vg)] = stream_state_NP( next_states_stream[(ug,vg)], head_movement[ug],next_channel_quality)
    t+=1

    
print("starting with the simulation phase")
t = 0
num_simulations = 50000
while t < num_simulations:
    channel_quality = cachin_states[ (reward_penalty_coeffs[0], l_max_range[0], user_gps[0], video_gps[0],1) ].CQ    
    next_states = {}
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    for h in h_range:
                        key = (p,l_m,ug,vg,h)
                        next_states[key] = cache_state_without_NP_and_CQ(Action_tables[key][cachin_states[key]], cachin_states[key], T = max_tiles(video_group = vg))
    for ug in user_gps:
        for vg in video_gps:
            next_states_stream[(ug,vg)] = stream_state_without_NP_and_CQ(streamin_states[(ug,vg)], video_group = vg)
    
    head_movement = {0 : np.random.choice(list(head_movement_lookup_table[0].keys()), p=list(head_movement_lookup_table[0].values()))}
    head_movement[1] = np.random.choice(list(head_movement_lookup_table[1].keys()), p=list(head_movement_lookup_table[1].values()))
    head_movement[2] = np.random.choice(list(head_movement_lookup_table[2].keys()), p=list(head_movement_lookup_table[2].values()))
    
    if t%5 == 0:
        next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    for h in h_range:
                        key = (p,l_m,ug,vg,h)
                        Cost_vec[key].append( Networking_Cost_CQ( Action_tables[key][cachin_states[key]], channel_quality))
                        Freeze_prob[key].append(Prob_screen_freeze(cachin_states[key], video_group = vg))                        
                        if h == 0:
                            cachin_states[key] = cache_state_NP( next_states[key], 1, next_channel_quality)
                        else:
                            cachin_states[key] = cache_state_NP( next_states[key], head_movement[ug], next_channel_quality)
                        if cachin_states[key].R == 1:
                            first_seg = cachin_states[key].n1  
                            next_action = Action_tables[key][cachin_states[key]]
                            if next_action[0] == 1:
                                first_seg +=next_action[1]
                                FOV_quality[key].append(FOV_quality_fn(first_seg, video_group = vg))
                        state_vec[key].append(cachin_states[key])
    for ug in user_gps:
        for vg in video_gps:
                    Cost_vec_stream[(ug,vg)].append( Networking_Cost_stream(streamin_states[(ug,vg)], next_states_stream[(ug,vg)]))
                    streamin_states[(ug,vg)] = stream_state_NP( next_states_stream[(ug,vg)], head_movement[ug],next_channel_quality)
                    if streamin_states[(ug,vg)].R == 1:
                        first_seg = streamin_states[(ug,vg)].n1
                        first_seg +=stream_action(streamin_states[(ug,vg)], video_group = vg)
                        FOV_quality_stream[(ug,vg)].append(FOV_quality_stream_fn(first_seg, video_group = vg))
                    Freeze_prob_stream[(ug,vg)].append(Prob_screen_freeze_stream(streamin_states[(ug,vg)], video_group = vg))
                    state_vec_stream[(ug,vg)].append(streamin_states[(ug,vg)])
    t+=1

fn = 0
var1 = Cost_vec
var1p = Cost_vec_stream
var2 = FOV_quality
var2p = FOV_quality_stream
fn+=1
save_sub_figure(fn,
      [       
       [ sum(var1[(p,5,1,1,1)])/len(var1[(p,5,1,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,1,1)])/len(var1[(p,0,1,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,1)])/len(var1p[(1,1)]) ]*len(reward_penalty_coeffs), 
       [ sum(var1[(p,5,1,1,0)])/len(var1[(p,5,1,1,0)]) for p in reward_penalty_coeffs]
      ], 
      [         
        [ sum(var2[(p,5,1,1,1)])/len(var2[(p,5,1,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,1,1)])/len(var2[(p,0,1,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,1)])/len(var2p[(1,1)]) ]* len(reward_penalty_coeffs), 
        [ sum(var2[(p,5,1,1,0)])/len(var2[(p,5,1,1,0)]) for p in reward_penalty_coeffs]
      ],
      [ "predictive-buffering", "short-term buffering", "no-buffering", "Benchmark"],
       "Resource usage", "FOV quality", file_name_suffix = "policies")
#