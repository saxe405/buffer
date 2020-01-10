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
M_temp = int(max_tiles(video_group = args.video_group)*args.Mtrust)
All_states = set()
T = max_tiles(video_group = args.video_group)
for n1 in range(T+1):
    for n2 in range(T+1):
        for R in range(Q+1):
            for CQ in range(CQ_matrix.shape[0]):
                for L in range(args.L_max+1):
                    n_states = set([new_state(n1,n2,L,0,R,CQ)])
                    if args.L_max > 0:
                        n_states = n_states.union(set([new_state(n1,n2,L,x,R,CQ) for x in range(1,M_temp)]))
                    All_states = All_states.union(n_states)
            
if args.verbosity:
    print("State space computed")

#Step 2: Initilization
policy = {}
utility = {}
for state in All_states:
    Valid_actions = [ x for x in All_actions if isValidAction(x,state) ]
    policy[state] = Valid_actions[np.random.choice([x for x in range(len(Valid_actions))])]
    utility[state] = np.random.uniform(10,100)
if args.verbosity:
    print("Initilization successful!")
delta_vec = []
debugging = True
#Step 3: Policy evaluation

start = time.time()
delta = 100
while delta > 0.1:
    delta = 0
    for state in All_states:
        temp = utility[state]
        neighbours = Neighbours(policy[state],state)
        utility[state] = 0
        for sp in neighbours:            
            prob_transition = Probability_matrix(policy[state],state,sp)            
            utility[state] += gamma*prob_transition*utility[sp]
        delta = max(delta, abs(temp-utility[state]))
        #if debugging:
        #    print(delta)
        delta_vec.append(delta)
if args.verbosity:
    print("Policy evaluated!")
end = time.time()
elapsed_time = end - start
if debugging:
    plt.style.use('seaborn')

    # create a color palette

    plt.figure(1)
    fig, ax = plt.subplots()
    x_vals = [(x*elapsed_time/len(delta_vec)) for x in range(len(delta_vec))]
    ax.plot(x_vals, delta_vec, linewidth = 1, color='blue', label = 'streaming',alpha=0.9 )
    #xtik = [ "" for x in range(len(delta_vec))]
    #xtik[0] = 0
    #xtik[int(len(delta_vec)/2)] = elapsed_time/2
    #xtik[-1] = elapsed_time
    #ax.set_xticks([0,int(len(x_vals)/2), len(x_vals)-1])
    ax.set_xticks([0,elapsed_time/2,elapsed_time])
    plt.xlabel("time (s)")
    plt.ylabel("Utility")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images/Utility_convergence.png")

if debugging:
    imp_state = new_state(2,3,0,2,4,2)

#Step 4: policy improvement
policy_stable = False
while policy_stable != True:
    policy_stable = True
    for state in All_states:
        #if debugging and state == imp_state:
            #print(policy[state])
        temp = policy[state]
        Valid_actions = [ x for x in All_actions if isValidAction(x,state) ]
        policy[state] = Valid_actions[0]
        util = Reward_matrix(policy[state],state) + sum([ gamma*utility[sp]*Probability_matrix(policy[state],state,sp) for sp in Neighbours(policy[state], state)])
        for action in Valid_actions:
            neighbours = Neighbours(action,state)
            this_util = Reward_matrix(action,state) + sum([ gamma*utility[sp]*Probability_matrix(action,state,sp) for sp in neighbours])
            if this_util > util:
                policy[state] = action
                util = this_util
         #       if debugging and state == imp_state:
                    #print(action, this_util)
        if temp != policy[state]:
            policy_stable = False

#Step 5: writing to disk
if args.verbosity:
    print("Writing Utilities to excel sheet")
book_name = Utility_book_name(args.penalty, args.L_max, args.user_group, args.video_group,args.pi0, args.Mtrust)
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