__all__ = ["Identity"]
from Definitions import *
def matrix_F():
	F_3 = np.zeros((Up.shape[0]*(M-1),Up.shape[0]*M))
	F_3 = np.block([
			[F_3],
			[Up, np.zeros((Up.shape[0],(M-1)*Up.shape[0]))]
		])
	F_2 = np.zeros((F_3.shape[0], Q*F_3.shape[0] ))
	for i in range(1,Q):
		F_2 = np.block([
			[F_2],
			[ np.zeros(( F_3.shape[0], (i-1)*F_3.shape[0])), F_3, np.zeros((F_3.shape[0], (Q-i)*F_3.shape[0] ))]
			])
	F_1 = np.block([
			[Identity(0,2)*F_2, np.zeros((F_2.shape[0], (T)*F_2.shape[0])) ]
		])
	for i in range(1,T+1):
		#print(F_1.shape)
		#print(i* F_2.shape[0] + F_2.shape[0] + (T-i)*F_2.shape[0])		
		F_1 = np.block([
			[F_1],
			[np.zeros(( F_2.shape[0], i*F_2.shape[0])), Identity(i,2)*F_2, np.zeros((F_2.shape[0], (T-i)*F_2.shape[0])) ]
			])
	F = np.block([
			[Identity(0,1)*F_1, np.zeros((F_1.shape[0], (T)*F_1.shape[0])) ]
		])
	for i in range(1,T+1):
		F = np.block([
			[F],
			[np.zeros(( F_1.shape[0], i*F_1.shape[0])), Identity(i,1)*F_1, np.zeros((F_1.shape[0], (T-i)*F_1.shape[0])) ]
			])
	return F		

def matrix_L():
	J_2p = np.block([
		[W, np.zeros((W.shape[0], W.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		J_2p = np.block([
			[J_2p],
			[ np.zeros((W.shape[0],W.shape[0]*i)), W, np.zeros((W.shape[0], W.shape[0]*(M-1-i)))]
			])
	J_1p = np.zeros((J_2p.shape[0], J_2p.shape[0]*Q))
	for i in range(1,Q):
		J_1p = np.block([
			[J_1p],
			[ np.zeros((J_2p.shape[0],J_2p.shape[0]*(i-1))), J_2p, np.zeros((J_2p.shape[0], J_2p.shape[0]*(Q-i)))]
			])
	J_1 = np.block([
		[J_1p, np.zeros((J_1p.shape[0], J_1p.shape[0]*(T)))]
		])
	for i in range(1,T+1):
		J_1 = np.block([
			[J_1],
			[ np.zeros((J_1p.shape[0],J_1p.shape[0]*i)), J_1p, np.zeros((J_1p.shape[0], J_1p.shape[0]*(T-i)))]
			])
	#second part
	J_3 = np.block([
		[np.zeros(Up.shape), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-2)))]
		])
	for i in range(1,M-1):
		J_3 = np.block([
			[J_3],
			[np.zeros(( Up.shape[0],Up.shape[0]*(i+1))), Up,  np.zeros(( Up.shape[0],Up.shape[0]*(M-i-2))) ]
			])
	J_2 = np.zeros((J_3.shape[0], J_3.shape[0]*Q))
	for i in range(1,Q):
		J_2 = np.block([
			[J_2],
			[ np.zeros((J_3.shape[0],J_3.shape[0]*(i-1))), J_3, np.zeros((J_3.shape[0], J_3.shape[0]*(Q-i)))]
			])
	L_3p = np.block([
		[Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		L_3p = np.block([
			[L_3p],
			[ np.zeros((Up.shape[0],Up.shape[0]*i)), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1-i)))]
			])
	L_2p = np.zeros((L_3p.shape[0], L_3p.shape[0]*Q))
	for i in range(1,Q):
		L_2p = np.block([
			[L_2p],
			[ np.zeros((L_3p.shape[0],J_3.shape[0]*(i-1))), L_3p, np.zeros((L_3p.shape[0], L_3p.shape[0]*(Q-i)))]
			])
	L_1p = np.block([
		[L_2p, np.zeros((L_2p.shape[0], L_2p.shape[0]*(T)))]
		])
	for i in range(1,T+1):
		L_1p = np.block([
			[L_1p],
			[ np.zeros((L_2p.shape[0],L_2p.shape[0]*i)), L_2p, np.zeros((L_2p.shape[0], L_2p.shape[0]*(T-i)))]
			])
	L_1 = np.block([
			[Identity(0,2)*J_2, (1-Identity(0,2))*L_2p, np.zeros((J_2.shape[0],J_2.shape[0]*(T-1)))]
		])
	for i in range(1,T):
		L_1 = np.block([
			[L_1],
			[ np.zeros((J_2.shape[0], J_2.shape[0]*i)), Identity(i,2)*J_2 , (1-Identity(i,2))*L_2p, np.zeros((J_2.shape[0], J_2.shape[0]*(T-i-1)))]
			])
	L_1 = np.block([
			[L_1],
			[np.zeros((J_2.shape[0], J_2.shape[0]*T)), Identity(T,2)*J_2]
		])
	L = np.block([
			[Identity(0,1)*L_1 + J_1, (1-Identity(0,1))*L_1p, np.zeros((J_1.shape[0],J_1.shape[0]*(T-1)))]
		])
	for i in range(1,T):
		L = np.block([
			[L],
			[ np.zeros((J_1.shape[0], J_1.shape[0]*i)), Identity(i,1)*L_1+J_1 , (1-Identity(i,1))*L_1p, np.zeros((J_2.shape[0], J_2.shape[0]*(T-i-1)))]
			])
	L = np.block([
			[L],
			[np.zeros((J_1.shape[0], J_1.shape[0]*T)), Identity(T,1)*L_1+J_1]
		])
	return L

def matrix_BIJ(C, j, index):
	#j,M - for 1,3,4
	#j+1,M - for 2
	row_num = j
	if index == 2:
		row_num+=1
	Bij = np.block([ 
		[np.zeros((C.shape[0]*(row_num), C.shape[0]*(T+1)))]
		[np.zeros((C.shape[0],C.shape[0]*(M-1))), C, np.zeros((C.shape[0],C.shape[0]*(T-M)))]
		[np.zeros((C.shape[0]*(T-row_num), C.shape[0]*(T+1)))]
		])
	return Bij


def matrix_B():
	C_6 = np.block([
		[W, np.zeros((W.shape[0], W.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		C_6 = np.block([
			[C_6],
			[np.zeros((W.shape[0], W.shape[0]*i)), W, np.zeros((W.shape[0], W.shape[0]*(M-i-1)))]
			])
	C_5 = np.block([
			[np.zeros((C_6.shape[0], (Q-1)*C_6.shape[0])), C_6],
			[np.zeros((C_6.shape[0]*(Q-1), C_6.shape[0]*Q))]
		])
	C_4 = np.block([
		[Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		C_4 = np.block([
			[C_4],
			[np.zeros((Up.shape[0], Up.shape[0]*i)), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-i-1)))]
			])
	C_3 = np.block([
			[np.zeros((C_4.shape[0], (Q-1)*C_4.shape[0])), C_4],
			[np.zeros((C_4.shape[0]*(Q-1), C_4.shape[0]*Q))]
		])		
	C_2 = np.zeros((Up.shape[0], M*Up.shape[0]))
	for i in range(1,M):
		C_2 = np.block([
			[np.zeros((Up.shape[0],(M-i)*Up.shape[0] )), Up, np.zeros((Up.shape[0], Up.shape[0]*(i-1)))],
			[C_2]
			])
	C_1 = np.block([
			[np.zeros((C_2.shape[0], (Q-1)*C_4.shape[0])), C_2],
			[np.zeros((C_2.shape[0]*(Q-1), C_2.shape[0]*Q))]
		])
	B = 0
	for i in range(T+1):
		this_row = Identity(i,1)*Identity(0,2)*matrix_BIJ(C_1,0,1) + Identity(i,1)*(1-Identity(0,2))*matrix_BIJ(C_3,0,2)+ (1-Identity(i,1))*matrix_BIJ(C_3,0,3) + matrix_BIJ(C_5,0,4)
		for j in range(1,T+1):
			new_block = Identity(i,1)*Identity(j,2)*matrix_BIJ(C_1,j,1) + Identity(i,1)*(1-Identity(j,2))*matrix_BIJ(C_3,j,2)+ (1-Identity(i,1))*matrix_BIJ(C_3,j,3) + matrix_BIJ(C_5,j,4)
			this_row = np.block([
				[this_row, new_block]
				])
		if i == 0:
			B = this_row
		else:
			B = np.block([
				[B],
				[this_row]
				])
	return B

def matrix_L0():
	return 0

matrix_F()
matrix_B()
matrix_L()
matrix_L0()