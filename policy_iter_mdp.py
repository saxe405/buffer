# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:20:46 2019

@author: sapoorv
"""

__all__ = ["Identity", "F","isValidAction", "isValidState"]
from Definitions import *
#__all__ = [ "Neighbours", "Probability_matrix"]
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from head_movements import *
from reward_rules import Reward_matrix
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import itertools
#Step 1: creating a list of all the states
#prev_set = set()
#All_states = set([starting_State_f(T = max_tiles(video_group = args.video_group))])
#added_members = set()
#last_added = set([starting_State_f(T = max_tiles(video_group = args.video_group))])
#while len(last_added) > 0:    
#    added_members = set()
#    for member in last_added:
#        for a in All_actions:            
#            added_members = added_members.union(Neighbours(a,member))
#    added_members = added_members - All_states # removing already existing members
#    last_added = added_members
    #print(len(added_members))
#    All_states = All_states.union(added_members)
print('starting state space computation')
M_temp = int(max_tiles(video_group = args.video_group)*args.Mtrust)
All_states = set()
T = max_tiles(video_group = args.video_group)

all_lists = [
        [n1 for n1 in range(T+1)],
        [n2 for n2 in range(T+1)],
        [L  for L in range(args.L_max+1)],
        [M  for M in range(M_temp) if args.L_max > 0 or M == 0],
        [R  for R  in range(1,Q+1)],
        [CQ for CQ in range(CQ_matrix.shape[0])]
        ]
for element in itertools.product(*all_lists):
    n_state = new_state(element[0], element[1], element[2], element[3], element[4], element[5])
    All_states.add(n_state)
           
if args.verbosity:
    print("State space computed")

#Step 2: Initilization
def get_list_elem(l, ind):
    return l[ind]
policy = dict(zip(All_states, [ get_list_elem( [ x for x in All_actions if isValidAction(x,state) ],
                                                 np.random.choice(len([ x for x in All_actions if isValidAction(x,state) ]))) for state in All_states
                                ]))
utility = dict(zip(All_states, list(np.random.uniform(-100,-10,len(All_states)))))
if args.verbosity:
    print("Initilization successful!")
delta_vec = []
debugging = False
#Step 3: Policy evaluation

policy_stable = False
num_repeats = 0

while policy_stable != True:
    print('Policy evaluation started (step num ' + str(num_repeats) + ' )')
    num_repeats +=1
    start = time.time()
    delta = 100
    
    delta_th = max( 10, 100/num_repeats)
    while delta > delta_th:
        delta = 0
        for state in All_states:
            temp = utility[state]
            
            neighbours = list(Neighbours(policy[state],state))
            utility[state] = Reward_matrix(policy[state], state)
            for sp in neighbours:            
                prob_transition = Probability_matrix(policy[state],state,sp)            
                utility[state] += gamma*prob_transition*utility[sp]
            delta = max(delta, abs(utility[state]-temp))

        if debugging:
            print(delta)
            delta_vec.append(delta)
    if args.verbosity:
        print("Policy evaluated!")
    end = time.time()
    elapsed_time = end - start
    

    imp_state = new_state(1,0,0,0,2,1)

    #Step 4: policy improvement

    print('Policy improvement started!')
    policy_stable = True
    count_weird_states = 0
   
    for state in All_states:
        if debugging and state == imp_state:
            #print(policy[state])
       
        temp = policy[state]
        Valid_actions = [ x for x in All_actions if isValidAction(x,state) ]

        neighs = list(Neighbours(policy[state], state))
        util = Reward_matrix(policy[state],state) + sum([ gamma*utility[sp]*Probability_matrix(policy[state],state,sp) for sp in neighs])
        for action in Valid_actions:
            neighs = Neighbours(action,state)
            this_util = Reward_matrix(action,state) + sum([ gamma*utility[sp]*Probability_matrix(action,state,sp) for sp in neighs])
            if this_util > util:
                policy[state] = action
                util = this_util

        if temp != policy[state]:
            policy_stable = False
            count_weird_states +=1
            if debugging and count_weird_states< 5:
                print(temp)
                print(policy[state])
                print(util)
                print(utility[state])
            
    if not policy_stable:
        print('policy not stable')
        if count_weird_states > 0 and count_weird_states < 2:
            print('less than 2 states with issue')
            policy_stable = True
    
    if not policy_stable:
        print('repeating evaluation!')
    print('Num weird states'  + str(count_weird_states))

print('policy found')
#Step 5: writing to disk
if args.verbosity:
    print("Writing Utilities to excel sheet")
book_name = Utility_book_name(args.penalty, args.L_max, args.user_group, args.video_group,args.pi0, args.Mtrust, args.head_move, args.NP_log)
book = xlsxwriter.Workbook(book_name)
worksheet = book.add_worksheet()

row = 0
for key in policy:
    data = [key.n1, key.n2, key.L, key.N, key.R, key.CQ]
    data.append(str(policy[key][0]) + "x" + str(policy[key][1]))
    for col in range(len(data)):
        worksheet.write(row, col, data[col])
    row+=1

if args.verbosity:
    print("Written to " + book_name)
book.close()