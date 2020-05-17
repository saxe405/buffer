from Definitions import *
from collections import defaultdict
import pandas as pd
theta_p0 = 1
theta_y0 = 1
theta_r0 = 1
theta_p_range = int(45/theta_p0)
theta_y_range = int(90/theta_y0)
theta_r_range = int(45/theta_r0)

def head_movement_probability_artificial(theta_p, theta_y, theta_r, user_group = args.user_group):                        # probability that the user moves his head with this much angle for each dimension (pitch, yaw, roll)
    prob_1 = 0.98
    prob_2 = 0.985
    prob_3 = 0.98
    if user_group == 1:
        prob_1 += 0.005
        prob_2 += 0.01
        prob_3 += 0.01
    elif user_group == 0:
        prob_1 += 0.01
        prob_2 += 0.01
        prob_3 += 0.02
    if theta_p != 0:
        prob_1 = (1-prob_1)/theta_p_range
    if theta_y != 0:
        prob_2 = (1-prob_2)/theta_y_range
    if theta_r != 0:
        prob_3 = (1-prob_3)/theta_r_range
    return prob_1*prob_2*prob_3



def head_movement_probability(theta_x, theta_y, theta_z, probability_distribution, user_group = args.user_group):
    k = '({x}, {y}, {z})'.format(x=theta_x, y=theta_y, z=theta_z)    
    return probability_distribution[k]
    

def compute_head_movement_table(npl = args.NP_log):
    file_contents = pd.read_csv('headmoves/'+ npl +'.csv')
    probability_distribution = defaultdict(lambda : 0)
    columns = file_contents.columns
    for ind,row in file_contents.iterrows():
        probability_distribution[row[columns[0]]] = row[columns[1]]
    print("Building the head movement lookup table: ")
    head_movement_lookup_table = {0: {}, 1:{}, 2:{}}
    head_movement_lookup_table[0] = { 1.0 : head_movement_probability(0,0,0,probability_distribution, user_group = 0)}
    head_movement_lookup_table[1] = { 1.0 : head_movement_probability(0,0,0,probability_distribution, user_group = 1)}
    head_movement_lookup_table[2] = { 1.0 : head_movement_probability(0,0,0,probability_distribution, user_group = 2)}

    for i in range(100):
        for ug in range(3):
            head_movement_lookup_table[ug][i/100] = 0

        
    for ug in range(3):    
        for i in range(1,theta_p_range):
            for j in range(1,theta_y_range):
                for k in range(1,theta_r_range):
                    ratio = round((1-i/theta_p_range)*(1-j/theta_y_range)*(1-k/theta_r_range),2)                
                    prob = head_movement_probability(i,j,k,probability_distribution, user_group = ug)                
                    head_movement_lookup_table[ug][ratio] += prob
        total = sum(head_movement_lookup_table[ug].values())    
        head_movement_lookup_table[ug] = {k: v/total for k, v in head_movement_lookup_table[ug].items()}
    return head_movement_lookup_table

head_movement_lookup_table = compute_head_movement_table(npl = args.NP_log)

def head_movement(num,den, user_group = args.user_group):
    if args.head_move == 0:
        if num == den:
            return 1
        return 0
    return sum([ v for k,v in head_movement_lookup_table[user_group].items() if k > round((num-1)/den,2) and k <= round(num/den,2) ])