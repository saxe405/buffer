import matplotlib.pyplot as plt
import re
import xlrd
import plotly as py
import plotly.graph_objs as go
__all__ = ["Identity", "Neighbours", "Probability_matrix"]
from Definitions import *
from transition_rules import Neighbours
from transition_rules import Probability_matrix
import seaborn as sns
import time
import numpy as np
from shutil import copyfile
from reward_rules import Networking_Cost_CQ
import os

def read_traces(trace_num):
	f=open("5G_traces/WiGig_"+ str(trace_num) + ".txt", "r")
	f.contents().split('\n')

def save_plot(plot_num, Performance_index, y_lab, x_lab = 'Time'):
	plt.figure(plot_num)
	fig, ax = plt.subplots()

	ax.plot([0.1*x for x in range(len(Performance_index))], Performance_index, 'b-', label = 'predictive caching' )
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.tight_layout()
	plt.grid(True)
	plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/"+ y_lab +".png")

data = xlrd.open_workbook('Updated Utilities_1.0.xlsx')
table = data.sheets()[0]
ncols = table.ncols
num_iterations = int((ncols-8)/2)

Action_table = {}

for i in range(table.nrows):
	vals = table.row_values(i)		
	s = new_state(int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]),int(vals[4]),int(vals[5]))
	a = (int(vals[ncols-1][0]), int(vals[ncols-1][2:]) )	
	Action_table[s] = a
print("Action table ready")
t = 0
state = starting_State
Cost_vec = [0]
QoE_vec = [0]
Network_util = [0]
print("Starting with the initial phase")
while t < 1000:
	t+=1
	best_action = Action_table[state]	
	Neighs = list(Neighbours(best_action, state))
	Probs = [ Probability_matrix(best_action,state, s2) for s2 in Neighs]
	if len(Probs) == 0:
		print(state, best_action, Neighs)
	new_state = Neighs[np.random.choice([x for x in range(0,len(Probs))], p=Probs)]
	
	Cost_vec.append( Cost_vec[-1]*gamma + Networking_Cost_CQ(best_action,state.CQ) )
	Network_util.append(gamma*Network_util[-1]+ Network_utilization(state,best_action))
	
	state = new_state

state_vec = [state]
FOV_vec = [FOV_available(state)]
Cost_vec = [Cost_vec[-1]]
QoE_vec = [QoE_vec[-1]]
Network_util = [Network_util[-1]]
FOV_hand = [ FOV_handover(state)]
Freeze_prob = [Prob_screen_freeze(state)]
print("starting with the simulation phase")
t = 0
while t < 200:
	t+=1
	best_action = Action_table[state]
	Neighs = list(Neighbours(best_action, state))
	Probs = [ Probability_matrix(best_action,state, s2) for s2 in Neighs]
	new_state = Neighs[np.random.choice([x for x in range(0,len(Probs))], p=Probs)]
	
	state_vec.append(new_state)
	FOV_vec.append(FOV_available(new_state))
	Cost_vec.append(gamma*Cost_vec[-1] + Networking_Cost_CQ(best_action,state.CQ))
	Network_util.append(gamma*Network_util[-1]+ Network_utilization(state,best_action))
	FOV_hand.append(FOV_handover(new_state))
	Freeze_prob.append(Prob_screen_freeze(new_state))
	state = new_state	

save_plot(1,FOV_vec, "FOV_available")
save_plot(2,Cost_vec, "Discounted cost")
save_plot(3, [ x.CQ for x in state_vec], "Channel Quality")
save_plot(4, Network_util, "Network Utility")
save_plot(5, FOV_hand, "FOV availability of next segment")
save_plot(6, Freeze_prob, "Probabity of freeze")
#in the long run, how much time was spent in a frozen screen state