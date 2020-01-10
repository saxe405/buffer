from Definitions import *
theta_p0 = 4
theta_y0 = 2
theta_r0 = 1
theta_p_range = int(110/theta_p0)
theta_y_range = int(20/theta_y0)
theta_r_range = int(10/theta_r0)

def head_movement_probability(theta_p, theta_y, theta_r, user_group = args.user_group):						# probability that the user moves his head with this much angle for each dimension (pitch, yaw, roll)
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

def head_movement_make_table(ratio_n, user_group = args.user_group):
	prob = 0
	for i in range(theta_p_range):
		for j in range(theta_y_range):
			for k in range(theta_r_range):
				if round((1-i*theta_p0/120)*(1-j*theta_y0/20)*(1-k*theta_r0/10),2) == ratio_n:
					prob += head_movement_probability(i*theta_p0,j*theta_y0,k*theta_r0, user_group = user_group)
	return prob

print("Building the head movement lookup table: ")
head_movement_lookup_table = {0: {}, 1:{}, 2:{}}
head_movement_lookup_table[0] = { 0 : head_movement_make_table(0, user_group = 0)}
head_movement_lookup_table[1] = { 0 : head_movement_make_table(0, user_group = 1)}
head_movement_lookup_table[2] = { 0 : head_movement_make_table(0, user_group = 2)}

for i in range(1,101):
    for ug in range(3):
        head_movement_lookup_table[ug][i/100] = head_movement_make_table(i/100, user_group = ug)

for ug in range(3):
    prob_sum = sum(head_movement_lookup_table[ug].values())
    # make sum of probability of head movement = 1
    for elem in head_movement_lookup_table[ug]:
        head_movement_lookup_table[ug][elem] /= prob_sum
#print(sum(head_movement_lookup_table.values()))

def head_movement(num,den, user_group = args.user_group):
	return sum([ v for k,v in head_movement_lookup_table[user_group].items() if k > round((num-1)/den,2) and k <= round(num/den,2) ])