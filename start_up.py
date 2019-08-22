theta_p0 = 4
theta_y0 = 2
theta_r0 = 1
theta_p_range = int(120/theta_p0)
theta_y_range = int(20/theta_y0)
theta_r_range = int(10/theta_r0)

def head_movement_probability(theta_p, theta_y, theta_r):						# probability that the user moves his head with this much angle for each dimension (pitch, yaw, roll)
	prob_1 = 0.85
	prob_2 = 0.90
	prob_3 = 0.98
	if theta_p != 0:
		prob_1 = (1-prob_1)/theta_p_range
	if theta_y != 0:
		prob_2 = (1-prob_2)/theta_y_range
	if theta_r != 0:
		prob_3 = (1-prob_3)/theta_r_range
	return prob_1*prob_2*prob_3

def head_movement(ratio_n):
	prob = 0
	for i in range(theta_p_range):
		for j in range(theta_y_range):
			for k in range(theta_r_range):
				if abs((1-i*theta_p0)*(1-j*theta_y0)*(1-k*theta_r0) - ratio_n) < 1.05*ratio_n:
					prob += head_movement_probability(i*theta_p0,j*theta_y0,k*theta_r0)
	return prob
	
from Definitons import *
print("Building the head movement lookup table: ")
head_movement_lookup_table = { 0 : head_movement(0)}
for i in range(1,100):
	head_movement_lookup_table[i/100] = head_movement(i/100)