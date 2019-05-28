__all__ = ["Identity", "F","isValidAction", "isValidState"]
from Definitions import *
#__all__ = [ "Neighbours", "Probability_matrix"]
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from reward_rules import Reward_matrix
import argparse

Depth_n = set()
Depth_n_plus = set()
Depth_n.add(starting_State)
depth = 0
Completed_states = set()

parser = argparse.ArgumentParser()
parser.add_argument("--num_iters", "-ni", help = "number of value iterations to perform", type  = int, default = 5 )
parser.add_argument("--verbosity", "-v", help = "increase output verbosity", type = bool, default = False)
args = parser.parse_args()

def value_iteration_state(s):
	global Utility_table, Action_table
	util = -1000	
	for action in All_actions:
		if not isValidAction(action,s):
			continue
		this_util = -1000
		for sp in Neighbours(action,s):
			prob_transition = Probability_matrix(action,s,sp)
			this_util = Reward_matrix(action,s) + gamma*prob_transition*Utility_table[sp]
		if this_util > util:
			Action_table[s] = action
			#print(s, action, util)
		util = max(util, this_util )
	return util

def value_iteration():
	global Depth_n, Completed_states, Depth_n_plus, Utility_table, depth
	if args.verbosity:
		print( "working on depth ", depth)
	depth +=1
	for st in Depth_n:
		if st in Completed_states:
			continue
		Completed_states.add(st)
		Utility_table[st] = value_iteration_state(st)
		for act in All_actions:
			Depth_n_plus = Depth_n_plus.union(Neighbours(act,st))
	Depth_n = Depth_n_plus - Completed_states
	Depth_n_plus = set()
	if len(Depth_n) > 0:
		value_iteration()
		
def value_iteration_k_times(k):
	global depth, Depth_n, Completed_states
	depth = 0
	for i in range(k):
		print("Doing the value iteration step: ", i+1)		
		Depth_n.add(starting_State)
		depth = 0
		Completed_states = set()
		value_iteration()

value_iteration_k_times(args.num_iters)

if args.verbosity:
	print("Writing Utilities to disk")

f = open("Updated Utilities.txt", "w")
for key in Utility_table:
	f.write( str(key)+ ":" + str(Utility_table[key]) + "," + str(Action_table[key])+ "\n")
f.close()
#starting_State = new_state(0,0,0,0,Q,2)
#print(Neighbours(2,starting_State))
