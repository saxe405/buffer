import matplotlib.pyplot as plt
import re
import xlrd
import plotly as py
import plotly.graph_objs as go
__all__ = ["Identity"]
from Definitions import *
import seaborn as sns
import time
import numpy as np
from shutil import copyfile
import os

data = xlrd.open_workbook('Utilities/Updated Utilities.xlsx')
table = data.sheets()[0]
ncols = table.ncols
num_iterations = int((ncols-8)/2)
#Actions = -1*np.ones((len(CQs),2*T+L_max+1))
#UTs = np.zeros((len(CQs),2*T+L_max+1))
Actions = -1*np.ones((len(CQs),100*T+10*T+L_max+1))
UTs = np.zeros((len(CQs),100*T+10*T+L_max+1))
for i in range(table.nrows):
	vals = table.row_values(i)
	if int(vals[ncols-1]) == -1:
		continue
	y = int(vals[5])
	#x = int(vals[0])
	x = int(vals[0]*100 + 10* vals[1]+ vals[2])
		#continue
#	if Identity(int(vals[0]),1):
#		x += int(vals[1]) #add n2 if block 1 has enough tiles
#		if Identity(int(vals[1]),2):
#			x += int(vals[2]) #add L if both n1 and n2 have enough tiles
	if Actions[y][x] >=0 and Actions[y][x] != int(vals[ncols-1]):
		print(Actions[y][x],vals[ncols-1], vals[0:5])
	Actions[y][x] = int(vals[ncols-1])
	UTs[y][x] = vals[ncols - 2 - num_iterations]
x_vals = []
Action_vals = [[] for y in range(len(CQs))]

#for y in range(len(CQs)):
#	for x in range(100*T+10*T+L_max+1):
#		if Actions[y][x] >= 0:
#			Action_vals[y].append(Actions[y][x])
#			x_vals.append(x)
trace = go.Heatmap(z=Actions,
                   x=[x for x in range(Actions.shape[1])],
                   y=[y for y in range(len(CQs))])
data=[trace]
py.offline.plot(data, image_filename='heatmap', image='svg')
time.sleep(1)

#copyfile('{}/{}.svg'.format(os.path.expanduser('~/Downloads'), 'heatmap'))
'''
plt.figure(1)
fig, ax = plt.subplots()


ax.plot(loads, sdv_red_l, 'r-', label = 'alpha_i = 1, approximation' )

ax.plot(loads, sdv_red_s, 'r*', label = 'alpha_i = 1, simulation' )
ax.plot(loads, sdv_blue_ap, 'b-',label = 'alpha_i = i/2c, approximation' )
ax.plot(loads, sdv_blue_s,'b*', label = 'alpha_i = i/2c, simulation' )
ax.plot(loads, sdv_green_ap, 'g-', label = 'alpha_i = 0, approximation' )
ax.plot(loads, sdv_green_s, 'g*', label = 'alpha_i = 0, simululation' )
ax.plot(loads, sdv_yellow_ap, 'y-', label = 'alpha_i = 0.5+i*i/(2*th*th), approximation' )
ax.plot(loads, sdv_yellow_s, 'y*',label = 'alpha_i = 0.5+i*i/(2*th*th), simulation' )

##plt.plot(loads,sdv_red_l, 'r-', loads, sdv_red_s, 'r*', loads, sdv_blue_ap, 'b-',loads, sdv_blue_s,'b*', loads, sdv_green_ap, 'g-', loads, sdv_green_s, 'g*', loads, sdv_yellow_ap, 'y-', loads, sdv_yellow_s, 'y*')
plt.xlabel('Buffer level')
plt.ylabel('Channel Quality')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('#00FFCC')
plt.tight_layout()
plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/best_actions.png")


data = np.random.randn(6, 6)
y = ["CQ. {}".format(i) for i in range(1, 6)]
x = ["Cycle {}".format(i) for i in range(1, 7)]

qrates = np.array([0, 11,12,21,22,31,32])
norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

im, _ = heatmap(data, y, x, ax=ax3,
                cmap=plt.get_cmap("PiYG", 7), norm=norm,
                cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                cbarlabel="Quality Rating")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=["red", "black"])
'''