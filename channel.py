import pandas as pd
import numpy as np
import math

p = 0.33
q = 0.2
r = 0.8
CQ_matrix = np.matrix([
	[p, 			15.0*(1-p)/43.0, 	28.0*(1-p)/43.0	],
	[(1-q)/2, 		q, 					(1-q)/2			],
	[24*(1-r)/43, 	19*(1-r)/43, 		r 				]
])

CQs = [ x for x in range(CQ_matrix.shape[0]) ]				#channel qualities total 5 for now, for each state we have a corresponding cost of downloading
Cost_CQ_base = [ -0.3 + 30*math.exp(-2.093*x) 	for x in range(len(CQs))] 

def max_tiles_in_CQ(CQ):
	if CQ == 0:
		return 0
	if CQ == 1:
		return 5
	return 10

def filtered_data(activities,environments, areas):
	data = pd.read_csv("C:/Users/sapoorv/Downloads/CODE/python/VR cache/LTE signal strength traces/dataset/03_outputs/results_dataForStatistics/dataframe.csv") 
	data = data[ data.activity != "biking" ]
	data = data[ data.activity != "running" ]
	data = data[ data.activity != "in a train" ]
	data = data[ data.environment != "metropolis"]
	
	data = data.replace(to_replace="village", value="rural")
	data = data.replace(to_replace="middle of nowhere", value ="rural")
	data = data.replace(to_replace = "I am in a big crowd", value = "People around me")
	data = data.replace(to_replace =  "I am in a small crowd", value =  "People around me" )
	data = data.replace(to_replace =  "There are a few people around me", value =  "People around me" )

	#all_activities = ['standing', 'walking', 'in a bus', 'in a car' ]
	#all_environments = ['rural', 'town', 'large town', 'city', 'large city']
	#all_areas = ['I am alone', 'People around me']

	data = data[ data.activity.isin(activities) ]
	data = data[data.environment.isin(environments)]
	data = data[data.area.isin(areas)]
	return data

def channel_matrix( collapse = False ):
	activities = [ 'standing']
	environments = ['town', 'large town', 'city', 'large city']
	areas = ['People around me']

	data = filtered_data(activities,environments,areas)
	
	min_rsrq = min(data.rsrq)
	Channel_matrix = np.zeros(shape=(max(data.rsrq)+1-min_rsrq,max(data.rsrq)+1-min_rsrq))

	old = data['rsrq'].iloc[0] - min_rsrq
	for row in data.rsrq[1:]:
		Channel_matrix[old][row-min_rsrq] += 1
		old = row-min_rsrq
	Channel_matrix_collapsed = Channel_matrix.copy()
	for i in range(Channel_matrix.shape[0]):
		row_sum = sum(Channel_matrix[i])
		if row_sum > 0:
			Channel_matrix[i] = Channel_matrix[i]/row_sum

	writer = pd.ExcelWriter("CQ_matrix.xlsx", engine = 'xlsxwriter')
	df = pd.DataFrame(Channel_matrix)
	df.to_excel(writer,sheet_name = "original")
	
	merge_len = 3 # make 3 consecutive states as one 0,1,2 become new 0 - 3,4,5 become new 1
	new_len = int(len(Channel_matrix_collapsed)/merge_len)
	new_mat = np.zeros((new_len,new_len))
	for i in range(new_len):
		for j in range(new_len):
			new_mat[i,j] = Channel_matrix_collapsed[i*merge_len:(i+1)*merge_len, j*merge_len:(j+1)*merge_len].sum()	
	Channel_matrix_collapsed = new_mat
	for i in range(Channel_matrix_collapsed.shape[0]):
		row_sum = sum(Channel_matrix_collapsed[i])
		if row_sum > 0:
			Channel_matrix_collapsed[i] = Channel_matrix_collapsed[i]/row_sum
	df2 = pd.DataFrame(Channel_matrix_collapsed)
	
	df2.to_excel(writer, sheet_name = "collapsed")
	writer.save()
	if collapse:
		return new_mat
	return Channel_matrix 

def read_from_disk(filename, collapse = False):
	sheet = "original"
	if collapse:
		sheet = "collapsed"
	mat = np.matrix(pd.read_excel(filename, sheet_name = sheet,index_col = 0))	
	return mat
#print(read_from_disk("CQ_matrix.xlsx", collapse = True))

