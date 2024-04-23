import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy.signal import filtfilt

#lowpass filter for data
def butterworth_lowpass_filter(data, cutoff, nyq, fs, order):
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog = False)
    y = filtfilt(b,a,data)
    return y

#windows directory of files
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"

#names of the means and standard error data files
filetypes = {
    1:"50_dynamic.csv",
    2:"50_static.csv",
    3:"310_dynamic.csv",
    4:"255_dynamic.csv"}

z_start = 0.35653
#convert csv data into variables in the form of a list of lists
dyn_50 = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
dyn_50 = np.delete(dyn_50,0,0)
stat_50 = genfromtxt(DIR+filetypes[2],delimiter=',',dtype=float)
stat_50 = np.delete(stat_50,0,0)
dyn_310 = genfromtxt(DIR+filetypes[3],delimiter=',',dtype=float)
dyn_310 = np.delete(dyn_310,0,0)
dyn_255 = genfromtxt(DIR+filetypes[4],delimiter=',',dtype=float)
dyn_255 = np.delete(dyn_255,0,0)

#pre-define variables
dyn_50_ntime = []
dyn_50_y = []
dyn_50_z = []

stat_50_ntime = []
stat_50_vntime = []
stat_50_y = []
stat_50_z = []
stat_50_yvel = []
stat_50_zvel = []

dyn_310_ntime = []
dyn_310_vntime = []
dyn_310_y = []
dyn_310_z = []
dyn_310_yvel = []
dyn_310_zvel = []


dyn_255_ntime = []
dyn_255_y = []
dyn_255_z = []

#loops for seperating data
i = 0
j = 0
#loop for 50 stat
while i<=len(stat_50)-1:
    stat_50_ntime.append(stat_50[i][7])
    stat_50_y.append(stat_50[i][5])
    stat_50_z.append(stat_50[i][6]-z_start)
    i = i + 1

#loop for 310 dyn
while j<=len(dyn_310)-1:
    dyn_310_ntime.append(dyn_310[j][7])
    dyn_310_y.append(dyn_310[j][5])
    dyn_310_z.append(dyn_310[j][6]-z_start)
    j = j + 1


#compute velocity for stat 50
k = 1
while k <= len(stat_50_y)-2:
    yvel = (stat_50_y[k+1]-stat_50_y[k-1])/0.01
    zvel = (stat_50_z[k+1]-stat_50_z[k-1])/0.01
    stat_50_yvel.append(yvel)
    stat_50_zvel.append(zvel)
    k = k + 1
    
#compute velocity for dyn 310
p = 1
while p <= len(dyn_310_y)-2:
    yvel = (dyn_310_y[p+1]-dyn_310_y[p-1])/0.01
    zvel = (dyn_310_z[p+1]-dyn_310_z[p-1])/0.01
    dyn_310_yvel.append(yvel)
    dyn_310_zvel.append(zvel)
    p = p + 1

#loops for getting velocity times
r = 1
n = 1
while r <= len(stat_50_yvel)-1:
    stat_50_vntime.append(stat_50_ntime[r])
    r = r + 1

while n <= len(dyn_310_yvel)-1:
    dyn_310_vntime.append(dyn_310_ntime[n])
    n = n + 1
stat_50_yvel = np.delete(stat_50_yvel,519,0)
stat_50_zvel = np.delete(stat_50_zvel,519,0)
dyn_310_yvel = np.delete(dyn_310_yvel,404,0)
dyn_310_zvel = np.delete(dyn_310_zvel,404,0)

#cicular trajectory
trajectory_y = []
trajectory_z = []
tim = -0.85
ntim = -2.05
ntime = []
while tim <= 1.6:
    yt = 0.33*np.cos(tim)-0.33
    zt = 0.33*np.sin(tim)
    trajectory_y.append(yt)
    trajectory_z.append(zt)
    ntime.append(ntim)
    tim = tim + 0.005
    ntim = ntim + 0.005

#desired motion in time and y-z
Des_S50 = np.radians(50)
Des_D310 = np.radians(310)
S50_start_point = stat_50_y[0]
D310_start_point = dyn_310_y[0]
S50_end_point = stat_50_y[1041]
D310_end_point = dyn_310_y[811]
#S50_liney = stat_50_y
#D310_liney = dyn_310_y
S50_liney = np.linspace(S50_start_point,S50_end_point,100)
D310_liney = np.linspace(D310_start_point,D310_end_point,100)

i = S50_start_point
S50_linez = [x+(x*np.tan(Des_S50))+0.23 for x in S50_liney]
D310_linez = [x+(x*np.tan(Des_D310))-0.1 for x in D310_liney]

#filter the data using lowpass filter
fs = 200 #Hz
T50S = 5.205 
T310D = 4.055
sig_cutoff = 2 #Hz

ny = 0.5*fs
n50 = int(T50S*fs)
n310 = int(T310D*fs)
order = 2

stat50vy_filter = butterworth_lowpass_filter(stat_50_yvel, sig_cutoff, ny, fs, order)
stat50vz_filter = butterworth_lowpass_filter(stat_50_zvel, sig_cutoff, ny, fs, order)
dyn310vy_filter = butterworth_lowpass_filter(dyn_310_yvel, sig_cutoff, ny, fs, order)
dyn310vz_filter = butterworth_lowpass_filter(dyn_310_zvel, sig_cutoff, ny, fs, order)

#stacked plots for participant response.
"""
#original paper figure
fig, axs = plt.subplots(2)

la1 = axs[0].plot(stat_50_ntime,stat_50_y, label = "Y-Position", linewidth = 2.0, color = 'r')
la2 = axs[0].plot(stat_50_ntime,stat_50_z, label = "Z-Position", linewidth = 2.0, color = 'b')
la3 = axs[0].plot(stat_50_vntime,stat_50_yvel, label = "Y-Velocity", linewidth=2.0, color='tab:pink')
la4 = axs[0].plot(stat_50_vntime,stat_50_zvel, label = "Z-Velocity", linewidth=2.0, color='tab:purple')
axs[0].axvline(x=0, linewidth = 10, color="tab:gray", alpha = 0.75)
axs[0].tick_params(axis = 'x', labelsize = 30)
axs[0].tick_params(axis = 'y', labelsize = 30)
#axs[0].set_xlabel("Time (sec)", fontsize = 30)
#axs[0].set_ylabel("Position (m)/Velocity (m/s)", fontsize = 30)
#axs[0].legend(loc = 'best', title = 'Response Data', title_fontsize = 25, fontsize = 20)

ra1 = axs[1].plot(dyn_310_ntime,dyn_310_y, label = "Y-Position", linewidth = 2.0, color = 'r')
ra2 = axs[1].plot(dyn_310_ntime,dyn_310_z, label = "Z-Position", linewidth = 2.0, color = 'b')
ra3 = axs[1].plot(dyn_310_vntime,dyn_310_yvel, label = "Y-Velocity", linewidth = 2.0, color = 'tab:pink')
ra4 = axs[1].plot(dyn_310_vntime,dyn_310_zvel, label = "Z-Velocity", linewidth = 2.0, color = 'tab:purple')
ra5 = axs[1].plot(ntime, trajectory_y, label = "Y-Reference", linewidth = 2.0, color = 'k')
ra6 = axs[1].plot(ntime, trajectory_z, label = "Z-Reference", linewidth = 2.0, color = 'tab:brown')
axs[1].tick_params(axis = 'x', labelsize = 30)
axs[1].tick_params(axis = 'y', labelsize = 30)
axs[1].legend(loc = 'best', title = 'Response Data', title_fontsize = 25, fontsize = 22)
axs[1].axvline(x=0, linewidth = 10, color="tab:gray", alpha = 0.75)
axs[1].set_xlabel("Time (sec)", fontsize = 30)

fig.text(0.06, 0.6, 'Position (m)/Velocity (m/s)', va='center', ha='center', rotation='vertical', fontsize=25)
plt.tight_layout()
plt.show()
"""

plt.rc('font', family = 'Arial')

#y-z plane plot using filtered data for static trial
fig2, ax2 = plt.subplots(ncols = 2, sharex = True, sharey = True)

qa1 = ax2[0].plot(stat_50_y,stat_50_z, label = "Static Position")
qa2 = ax2[0].plot(S50_liney,S50_linez, label = "Static Reference")
ax2[0].set_ylabel("Z-Position (m)", fontsize = 30)
ax2[0].set_xlabel("Y-Position (m)", fontsize = 30)
ax2[0].set_box_aspect(1)
ax2[0].tick_params(axis = 'x', labelsize = 16)
ax2[0].tick_params(axis = 'y', labelsize = 16)

wa1 = ax2[1].plot(stat50vy_filter,stat50vz_filter, label = "Static Velocity")
ax2[1].set_xlabel("Y-Velocity (m/s)", fontsize = 30)
ax2[1].set_ylabel("Z-Velocity (m/s)", fontsize = 30)
ax2[1].set_box_aspect(1)
ax2[1].tick_params(axis = 'x', labelsize = 16)
ax2[1].tick_params(axis = 'y', labelsize = 16)

plt.show()

#y-z plane plot using filtered data for dynamic trial
fig3, ax3 = plt.subplots(ncols = 2, sharex = True, sharey = True)

ea1 = ax3[0].plot(dyn_310_y,dyn_310_z, label = "Dynamic Position")
ea2 = ax3[0].plot(D310_liney,D310_linez, label= "Dynamic Reference")
ax3[0].set_ylabel("Z-Position (m)", fontsize = 30)
ax3[0].set_xlabel("Y-Position (m)", fontsize = 30)
ax3[0].set_box_aspect(1)
ax3[0].tick_params(axis = 'x', labelsize = 16)
ax3[0].tick_params(axis = 'y', labelsize = 16)

ra1 = ax3[1].plot(dyn310vy_filter,dyn310vz_filter, label = "Dynamic Velocity")
ax3[1].set_xlabel("Y-Velocity (m/s)", fontsize = 30)
ax3[1].set_ylabel("Z-Velocity (m/s)", fontsize = 30)
ax3[1].set_box_aspect(1)
ax3[1].tick_params(axis = 'x', labelsize = 16)
ax3[1].tick_params(axis = 'y', labelsize = 16)

plt.show()


#original figure using filter data in plots
fig4, axs4 = plt.subplots(2)

la1 = axs4[0].plot(stat_50_ntime,stat_50_y, label = "Y-Position", linewidth = 2.0, color = 'r')
la2 = axs4[0].plot(stat_50_ntime,stat_50_z, label = "Z-Position", linewidth = 2.0, color = 'b')
la3 = axs4[0].plot(stat_50_vntime,stat50vy_filter, label = "Y-Velocity", linewidth=2.0, color='tab:pink')
la4 = axs4[0].plot(stat_50_vntime,stat50vz_filter, label = "Z-Velocity", linewidth=2.0, color='tab:purple')
axs4[0].axvline(x=0, linewidth = 10, color="tab:gray", alpha = 0.75)
axs4[0].tick_params(axis = 'x', labelsize = 20)
axs4[0].tick_params(axis = 'y', labelsize = 20)
#axs4[0].set_xlabel("Time (sec)", fontsize = 30)
#axs4[0].set_ylabel("Position (m)/Velocity (m/s)", fontsize = 30)
#axs4[0].legend(loc = 'best', title = 'Response Data', title_fontsize = 25, fontsize = 20)

ra1 = axs4[1].plot(dyn_310_ntime,dyn_310_y, label = "Y-Position", linewidth = 2.0, color = 'r')
ra2 = axs4[1].plot(dyn_310_ntime,dyn_310_z, label = "Z-Position", linewidth = 2.0, color = 'b')
ra3 = axs4[1].plot(dyn_310_vntime,dyn310vy_filter, label = "Y-Velocity", linewidth = 2.0, color = 'tab:pink')
ra4 = axs4[1].plot(dyn_310_vntime,dyn310vz_filter, label = "Z-Velocity", linewidth = 2.0, color = 'tab:purple')
ra5 = axs4[1].plot(ntime, trajectory_y, label = "Y-Reference", linewidth = 2.0, color = 'k')
ra6 = axs4[1].plot(ntime, trajectory_z, label = "Z-Reference", linewidth = 2.0, color = 'tab:brown')
#axs4[1].legend(loc = 'best', title = 'Response Data', title_fontsize = 25, fontsize = 20, framealpha = 1.0)
axs4[1].axvline(x=0, linewidth = 10, color="tab:gray", alpha = 0.75)
axs4[1].tick_params(axis = 'x', labelsize = 20)
axs4[1].tick_params(axis = 'y', labelsize = 20)
axs4[1].set_xlabel("Time (sec)", fontsize = 30)
fig4.text(0.06, 0.6, 'Position (m)/Velocity (m/s)', va='center', ha='center', rotation='vertical', fontsize=25)

plt.tight_layout()
plt.show()
