from Definitions import *
from common_analysis_fns import *

def save_plot(plot_num, Performance_index, y_lab, file_name, x_lab = 'Time'):
	# style
	plt.style.use('seaborn')
 
	# create a color palette
	palette = plt.get_cmap('Set1')
	plt.figure(plot_num)
	fig, ax = plt.subplots()
	length = len(Performance_index["streaming"])
	ax.plot([0.1*x for x in range(length)], Performance_index["streaming"], linewidth = 1, color=palette(1), label = 'streaming',alpha=0.9 )
	ax.plot([0.1*x for x in range(length)], Performance_index["only_short_predictions"], linewidth = 1, color=palette(2), label = 'short pred caching', alpha=0.9 )	
	ax.plot([0.1*x for x in range(length)], Performance_index["both_predictions"], linewidth = 1, color=palette(3), label = 's+l pred caching', alpha=0.9)		

	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
	#legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
	#legend.get_frame().set_facecolor('#00FFCC')
	plt.tight_layout()
	plt.grid(True)
	if args.trace_analysis:
		file_name += "PA"
	plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images/"+ file_name +".png")

if args.trace_analysis:
	print("reading 5G traces")
	Channel_path = read_traces(args.trace_number)


all_scenarios = ["streaming", "only_short_predictions", "both_predictions"]
Action_table = {}
Action_table["only_short_predictions"] = Action_table_builder('Updated Utilities_short_term_1.0.xlsx')
Action_table[ "both_predictions"] = Action_table_builder('Updated Utilities_1.0.xlsx')

print("Action tables ready")
t = 0

cachin_states = { "only_short_predictions" : starting_State}
cachin_states["both_predictions"] = starting_State
cachin_states["streaming" ] = s_state(T,Q,starting_State.CQ)

Cost_vec = { "streaming" : [0] , "only_short_predictions" : [0], "both_predictions" : [0]}
Network_util = { "streaming" : [0] , "only_short_predictions" : [0], "both_predictions" : [0]}

best_actions = { "only_short_predictions" : best_action_from_tables(Action_table, cachin_states, "only_short_predictions"), 
				 "both_predictions" : best_action_from_tables(Action_table, cachin_states, "both_predictions")
				}
print("Starting with the initial phase")
while t < 1000:
	t+=1

	channel_quality = cachin_states[ "both_predictions" ].CQ
	next_states = {}
	for scenario in all_scenarios:
		next_states[scenario] = cache_state_local(scenario, cachin_states, Action_table)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
	next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

	for scenario in all_scenarios:		
		Cost_vec[scenario].append( Cost_vec[scenario][-1]*gamma + Networking_Cost(scenario, cachin_states, next_states, Action_table) )		
		Network_util[scenario].append( Network_util[scenario][-1]*0.8 + Network_Utilization_local(scenario,cachin_states, next_states, Action_table) )
		
		#modify next states by applying the head movement transformations and channel quality variation
		cachin_states[scenario] = next_state_local(scenario, head_movement, next_channel_quality, next_states)

state_vec 	= { "streaming" : [] , "only_short_predictions" : [], "both_predictions" : []}

FOV_vec 	= {}
FOV_hand    = {}
Freeze_prob = {}
for scenario in all_scenarios:
	FOV_vec[ scenario] = [ FOV_available_local(scenario, cachin_states)]
	Cost_vec[scenario] = [Cost_vec[scenario][-1]]
#	QoE_vec[scenario]  = [QoE_vec[scenario][-1]]
	Network_util[scenario]  = [Network_util[scenario][-1]]
	FOV_hand[scenario] 		= [	FOV_handover_local(scenario, cachin_states) ] 
	Freeze_prob[scenario] 	= [	Freeze_prob_local(scenario, cachin_states) ]				

print("starting with the simulation phase")
t = 0
while t < 500:
	t+=1

	channel_quality = cachin_states[ "both_predictions" ].CQ
	next_states = {}
	for scenario in all_scenarios:
		next_states[scenario] = cache_state_local(scenario, cachin_states, Action_table)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
	next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )	
	
	for scenario in all_scenarios:
		Cost_vec[scenario].append( Cost_vec[scenario][-1]*gamma + Networking_Cost(scenario, cachin_states, next_states, Action_table) )		
		Network_util[scenario].append( Network_util[scenario][-1]*0.8 + Network_Utilization_local(scenario,cachin_states, next_states, Action_table) )
		FOV_hand[scenario].append(FOV_handover_local(scenario, cachin_states) )
		Freeze_prob[scenario].append(Freeze_prob_local(scenario, cachin_states))	
		FOV_vec[scenario].append(FOV_available_local(scenario, cachin_states))
		
		#modify next states by applying the head movement transformations and channel quality variation
		cachin_states[scenario] = next_state_local(scenario, head_movement, next_channel_quality, next_states)
		state_vec[scenario].append(cachin_states[scenario])
		
save_plot(1,FOV_vec, "FOV length available (segments))", "FOV_len_available" )
save_plot(2,Cost_vec, "Avg Networking Cost", "Average_networking_cost")
#save_plot(3, [ [x[0].CQ,x[1].CQ, x[2].CQ] for x in state_vec], "Channel Quality", "channel_quality")
save_plot(4, Network_util, "Network Utilization", "Network_Utilization" )
save_plot(5, FOV_hand, "Pr( FOV of next segment available)", "FOV_availability")
save_plot(6, Freeze_prob, "Probabity of freeze","Probabity_of_freeze")