__all__ = ["Identity", "F","isValidAction", "isValidState"]
from Definitions import *
#__all__ = [ "Neighbours", "Probability_matrix"]
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from head_movements import *
from reward_rules import Reward_matrix
import xlsxwriter

Depth_n = set()
Depth_n_plus = set()
Depth_n.add(starting_State)
depth = 0
Completed_states = set()

def value_iteration_state(s):
	global Utility_table, Action_table
	util = -100000	
	#print(s)	
	for action in All_actions:
		reward = Reward_matrix(action,s)
		
		total_prob = 0
		if not isValidAction(action,s):
			continue
		this_util = reward
		neighbours = Neighbours(action,s)	
		#print(neighbours)	
		
		for sp in neighbours:
			prob_transition = Probability_matrix(action,s,sp)			
			total_prob +=prob_transition
			this_util += gamma*prob_transition*Utility_table[sp]

		if total_prob == 0:   #just means that the action is not feasible in that state
			continue

		if this_util > util:
			Action_table[s] = action
			util = this_util
		
		#print(" Action chosen ", action, " with reward ", reward, " additional ", this_util - reward, " total ", this_util)
		
		#print(s, action, util)
			#util = max(util, this_util )		
		# 0 total prob just means that you can not get out of that state using a given action
		if total_prob!=0 and total_prob < 0.98:
			print(total_prob, s, action)
			neighbours = sorted(neighbours, key = lambda x: getattr(x, "n1") )
			for sp in neighbours:
				print(sp, Probability_matrix(action,s,sp))
			raise ValueError('Net probability out of state.')
	#print("action chosen ", Action_table[s])		
	Action_table_iterations[s] += "," + str(Action_table[s][0])+ "x" +str(Action_table[s][1])
	
	if util == -100000:
		print(s)
		print(total_prob)
		return -1000  #no point of returning a very high value which was acutually intended only to take maximum easily
	return util

def value_iteration():
	global Depth_n, Completed_states, Depth_n_plus, Utility_table, depth
	if args.verbosity:
		print( "working on depth ", depth)
	depth +=1
	if depth > 30:
		print(Depth_n)
	for st in Depth_n:
		if st in Completed_states:
			continue
		Completed_states.add(st)
		Utility_table[st] = value_iteration_state(st)
		Utility_table_iterations[st] += ","+str(Utility_table[st])
		for act in All_actions:
			Depth_n_plus = Depth_n_plus.union(Neighbours(act,st))
	Depth_n = Depth_n_plus - Completed_states
	Depth_n_plus = set()
	if len(Depth_n) > 0:
		value_iteration()

def write_to_disk():
	if args.verbosity:
		print("Writing Utilities to excel sheet")

	book = xlsxwriter.Workbook("Updated Utilities_"+ str(args.penalty)+".xlsx")
	worksheet = book.add_worksheet()

	#f = open("Updated Utilities.txt", "w")
	row = 0
	for key in Utility_table:	
		data = [key.n1, key.n2, key.L, key.N, key.R, key.CQ]
		data += Utility_table_iterations[key].split(',')
		data += Action_table_iterations[key].split(',')
		for col in range(len(data)):
			worksheet.write(row,col, data[col])	
		row+=1
	#	f.write( str(key)+ ":" + str(Utility_table[key]) + "," + str(Action_table[key])+ "\n")
	#new_state = namedtuple("State", "n1 n2 L N R CQ")
	book.close()
		
def value_iteration_k_times(k):
	global depth, Depth_n, Completed_states
	depth = 0
	for i in range(k):
		print("Doing the value iteration step: ", i+1)		
		Depth_n.add(starting_State)
		depth = 0
		Completed_states = set()
		value_iteration()

'''
print("The transition in focus happens with probability")
#print(isValidAction((0,0), new_state(9,0,1,0,4,0)))
print(Neighbours((0,0), new_state(10,0,1,1,1,0)))
print( Probability_matrix( (0,0), new_state(10,0,1,1,1,0), new_state(0,3,0,1,5,0)) )
print( Probability_matrix( (0,0), new_state(10,0,1,1,1,0), new_state(0,3,0,1,5,1)) )
print( Probability_matrix( (0,0), new_state(10,0,1,1,1,0), new_state(0,3,0,1,5,2)) )
'''
value_iteration_k_times(args.num_iters)
write_to_disk()
'''
action = 0
neighs = Neighbours(action, starting_State)

print("From state ", starting_State)
total = 0
for neigh in neighs:
	prob_transition = Probability_matrix(action, starting_State, neigh)
	print("to state ", neigh, " with probability ", prob_transition)
	total += prob_transition
print("Total out probability ", total)
'''
#starting_State = new_state(0,0,0,0,Q,2)
#print(Neighbours(2,starting_State))
