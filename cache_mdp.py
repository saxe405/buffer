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
Depth_n.add(starting_State_f())
depth = 0
Completed_states = set()

def value_iteration_state(s):
    global Utility_table, Action_table
    util = -100000    
    for action in All_actions:
        reward = Reward_matrix(action,s)
        
        total_prob = 0
        if not isValidAction(action,s):
            continue
        this_util = reward
        neighbours = Neighbours(action,s)    
            
        for sp in neighbours:            
            prob_transition = Probability_matrix(action,s,sp)            
            total_prob +=prob_transition
            this_util += gamma*prob_transition*Utility_table[sp]
        if s.n1 == 1 and s.n2 == 0 and s.L == 0 and s.N ==0 and s.R ==2 and s.CQ ==1:
                print(s, action, this_util)
        if total_prob == 0:   #just means that the action is not feasible in that state
            continue

        if this_util > util:
            Action_table[s] = action
            util = this_util
        
        if total_prob!=0 and total_prob < 0.98:
            print(total_prob, s, action)
            neighbours = sorted(neighbours, key = lambda x: getattr(x, "n1") )
            for sp in neighbours:
                print(sp, Probability_matrix(action,s,sp))
            raise ValueError('Net probability out of state.')    
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

def value_iteration_list(remaining_states):
    if args.verbosity:
        print("working on leftover states: ", str(len(remaining_states)))
    for st in remaining_states:
        Utility_table[st] = value_iteration_state(st)
        Utility_table_iterations[st] += ","+str(Utility_table[st])

def write_to_disk():
    if args.verbosity:
        print("Writing Utilities to excel sheet")
    book_name = Utility_book_name(args.penalty, args.L_max, args.user_group, args.video_group)
    book = xlsxwriter.Workbook(book_name)
    worksheet = book.add_worksheet()
    
    row = 0
    for key in Utility_table:
        data = [key.n1, key.n2, key.L, key.N, key.R, key.CQ]
        data += Utility_table_iterations[key].split(',')
        data += Action_table_iterations[key].split(',')
        for col in range(len(data)):
            worksheet.write(row,col, data[col])
        row+=1
    book.close()
        
def value_iteration_k_times(k, All_states):
    global depth, Depth_n, Completed_states
    depth = 0
    for i in range(k):
        print("Doing the value iteration step: ", i+1)
        ss = starting_State_f()
        Depth_n.add(ss)
        depth = 0
        Completed_states = set()
        value_iteration()
        value_iteration_list(All_states - Completed_states)

prev_set = set()
current_set = set([starting_State_f()])
added_members = set([starting_State_f()])
while len(added_members) > 0:    
    added_members = set()
    for member in current_set:
        for a in All_actions:            
            added_members = added_members.union(Neighbours(a,member))
    added_members = added_members - current_set # removing already existing members
    current_set = current_set.union(added_members)

value_iteration_k_times(args.num_iters, current_set)
write_to_disk()
