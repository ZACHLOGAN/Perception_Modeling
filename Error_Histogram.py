from matplotlib import ticker
import numpy as np
import csv
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter

#file directory for data files
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
#all filetypes
filetypes = {
	 1:"clean_alldata.csv"}
#import data and remove headers and subject numbers
Data = np.genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
Data = np.delete(Data,0,0)
Data = np.delete(Data,0,1)

#intialize variables for sorting
Angles = []
Response = []
Stat_Angles = []
Stat_Response = []
Dyn_Angles = []
Dyn_Response = []

Speed_Commands = []
Speed_Response = []
Speed_Errors = []
Static_Speed_C = []
Dynamic_Speed_C = []
Static_Speed_R = []
Dynamic_Speed_R = []
Static_Speed_E = []
Dynamic_Speed_E = []
i = 0
while i <= len(Data)-1:
    Angles.append(Data[i][1])
    Response.append(Data[i][5])
    Speed_Commands.append(Data[i][2])
    Speed_Response.append(Data[i][6])
    Speed_Errors.append(Data[i][10])
    if (Data[i][0] == 0):
        Stat_Angles.append(Data[i][1])
        Stat_Response.append(Data[i][5])
        Static_Speed_C.append(Data[i][2])
        Static_Speed_R.append(Data[i][6])
        Static_Speed_E.append(Data[i][10])
    elif (Data[i][0] == 1):
        Dyn_Angles.append(Data[i][1])
        Dyn_Response.append(Data[i][5])
        Dynamic_Speed_C.append(Data[i][2])
        Dynamic_Speed_R.append(Data[i][6])
        Dynamic_Speed_E.append(Data[i][10])
    i = i + 1

#compute the error for the three data sets
A_Error = []
S_Error = []
D_Error = []

i = 0
while i <= len(Angles)-1:
    Error = Response[i] - Angles[i]
    if (Error > 180 or Error < -180):

        if (Angles[i] < 180):
            A_Error.append( Response[i] - (360 + Angles[i]))
        elif (Angles[i] > 180):
            A_Error.append((360 + Response[i]) - (Angles[i]))

    else:
        A_Error.append( Error )

    i = i + 1

i = 0
while i <= len(Stat_Angles)-1:
    Error = Stat_Response[i] - Stat_Angles[i]
    if (Error > 180 or Error < -180):
        if (Stat_Angles[i] <= 180):
            S_Error.append( Stat_Response[i] - (360 + Stat_Angles[i]))
        elif (Stat_Angles[i] > 180):
            S_Error.append((360+Stat_Response[i]) - Stat_Angles[i]) 
    else:
        S_Error.append( Error )

    i = i + 1

i = 0
while i <= len(Dyn_Angles)-1:
    Error = Dyn_Response[i] - Dyn_Angles[i]
    if (Error > 180 or Error < -180):

        if (Dyn_Angles[i] < 180):
            err = Dyn_Response[i] - (360 + Dyn_Angles[i])
            D_Error.append( err )
        elif (Dyn_Angles[i] > 180):
            err = (360 + Dyn_Response[i]) - (Dyn_Angles[i])
            D_Error.append( err ) 

    else:
        D_Error.append( Error )

    i = i + 1


full_data = []
k = 0
while k <= len(A_Error)-1:
    vals = [Angles[k], Response[k], A_Error[k]]
    full_data.append(vals)
    k = k + 1

static_data = []
k = 0
while k <= len(S_Error)-1:
    vals = [Stat_Angles[k], Stat_Response[k], S_Error[k]]
    static_data.append(vals)
    k = k + 1

dynamic_data = []
k = 0
while k <= len(D_Error)-1:
    vals = [Dyn_Angles[k], Dyn_Response[k], D_Error[k]]
    dynamic_data.append(vals)
    k = k + 1

print(np.mean(A_Error), np.mean(S_Error), np.mean(D_Error))
print("\n")
print(np.std(A_Error), np.std(S_Error), np.std(D_Error))
print("\n")

speed_data = []
k = 0
while k<=len(Speed_Response)-1:
    vals = [Speed_Commands[k], Speed_Response[k], Speed_Errors[k]]
    speed_data.append(vals)
    k = k+1

speeds = [0.25, 0.5, 0.75, 1.0]
given_angles = [15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345]

A1 = np.array([[x,Angles.count(x)] for x in set(Angles)])
A1x = A1[:,0]
A1y = A1[:,1]

S1 = np.array([[x,Stat_Angles.count(x)] for x in set(Stat_Angles)])
S1x = S1[:,0]
S1y = S1[:,1]

D1 = np.array([[x,Dyn_Angles.count(x)] for x in set(Dyn_Angles)])
D1x = D1[:,0]
D1y = D1[:,1]

a_means = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
my_sum = 0
my_std = 0
counter = 0
for first in range(0, len(given_angles)-1, 1):
    my_sum = 0
    my_std = 0
    counter = 0
    for value in full_data:
        if value[0] == given_angles[first]:
            my_sum = my_sum + value[2]
            counter = counter + 1
    a_means[first] = my_sum/counter
    for value in full_data:
        if value[0] == given_angles[first]:
            my_std = my_std + (value[2]-my_sum/counter)**2
    a_std[first] = np.sqrt(my_std/counter)/counter

s_means = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
my_sum = 0
my_std = 0
counter = 0
for first in range(0, len(given_angles)-1, 1):
    my_sum = 0
    my_std = 0
    counter = 0
    for value in static_data:
        if value[0] == given_angles[first]:
            my_sum = my_sum + value[2]
            counter = counter + 1
    s_means[first] = my_sum/counter
    for value in static_data:
        if value[0] == given_angles[first]:
            my_std = my_std + (value[2]-my_sum/counter)**2
    s_std[first] = np.sqrt(my_std/counter)/counter

d_means = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
d_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
my_sum = 0
my_std = 0
counter = 0
for first in range(0, len(given_angles)-1, 1):
    my_sum = 0
    my_std = 0
    counter = 0
    for value in dynamic_data:
        if value[0] == given_angles[first]:
            my_sum = my_sum + value[2]
            counter = counter + 1
    d_means[first] = my_sum/counter
    for value in dynamic_data:
        if value[0] == given_angles[first]:
            my_std = my_std + (value[2]-my_sum/counter)**2
    d_std[first] = np.sqrt(my_std/counter)/counter


#check the speed error values
speed_means = [0, 0, 0, 0]
my_sum = 0
counter = 0
for first in range(0, len(speeds),1):
    my_sum = 0
    counter = 0
    print(first)
    for value in speed_data:
        if value[0] == speeds[first]:
            my_sum = my_sum + value[1]
            counter = counter + 1
    speed_means[first] = my_sum/counter

print(speed_means)

a_summary = []
s_summary = []
d_summary = []
full_summary = []
tracker = 0
while tracker <= len(a_means)-1:
    a_summary.append([a_means[tracker], a_std[tracker]])
    s_summary.append([s_means[tracker], s_std[tracker]])
    d_summary.append([d_means[tracker], d_std[tracker]])
    full_summary.append([given_angles[tracker], a_means[tracker], a_std[tracker], s_means[tracker], s_std[tracker], d_means[tracker], d_std[tracker]])
    tracker = tracker + 1

labels = ["Given_Angle", "All_Mean", "All_SD", "Stat_Mean", "Stat_SD", "Dyn_Mean", "Dyn_SD"]
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
name = DIR+"perception_errors.csv"
with open(name, 'w') as csvfile:
   testwriter = csv.writer(csvfile, delimiter = ',')
   testwriter.writerow(labels)
with open(name, 'a') as csvfile:
   testwriter = csv.writer(csvfile, delimiter = ',')
   testwriter.writerows(full_summary)

data_check = []
i = 0
while i <= len(Data)-1:
    data_check.append( [Data[i][1], Data[i][5], Data[i][7], Data[i][9], A_Error[i]] )
    i = i + 1

"""
name = DIR+"total_errors.csv"
with open(name, 'a') as csvfile:
   testwriter = csv.writer(csvfile, delimiter = ',')
   testwriter.writerows(data_check)
"""

#create three histograms of the data
tick_spacing = 10

fig1, ax1 = plt.subplots(1,1)
ax1.hist(A_Error, bins = 16)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax1.tick_params(axis = 'x', labelsize = 8)
ax1.tick_params(axis = 'y', labelsize = 8)
ax1.set_xlabel("Angle (Degrees)", fontsize = 12)
ax1.set_ylabel("Number of Responses", fontsize = 12)
ax1.grid()
plt.title("All Trial Errors", fontsize = 16)
plt.show()

fig2, ax2 = plt.subplots(1,1)
ax2.hist(S_Error, bins = 16)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.tick_params(axis = 'x', labelsize = 8)
ax2.tick_params(axis = 'y', labelsize = 8)
ax2.set_xlabel("Angle (Degrees)", fontsize = 12)
ax2.set_ylabel("Number of Responses", fontsize = 12)
ax2.grid()
plt.title("Only Static Trial Errors", fontsize = 16)
plt.show()

fig3, ax3 = plt.subplots(1,1)
ax3.hist(D_Error, bins = 16)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax3.tick_params(axis = 'x', labelsize = 8)
ax3.tick_params(axis = 'y', labelsize = 8)
ax3.set_xlabel("Angle (Degrees)", fontsize = 12)
ax3.set_ylabel("Number of Responses", fontsize = 12)
ax3.grid()
plt.title("Only Dynamic Trial Errors", fontsize = 16)
plt.show()

fig4, ax4 = plt.subplots(1,1)
ax4.hist(Response, bins = 16, label = 'Response', alpha = 0.8, color = 'red', edgecolor = 'red')
ax4.bar(A1x, A1y, color = 'blue', width = 15, alpha = 0.5)
ax4.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax4.tick_params(axis = 'x', labelsize = 8)
ax4.tick_params(axis = 'y', labelsize = 8)
ax4.set_xlabel("Angle (Degrees)", fontsize = 12)
ax4.set_ylabel("Number of Responses", fontsize = 12)
ax4.grid()
plt.title("All Trial Responses", fontsize = 16)
plt.show()

fig5, ax5 = plt.subplots(1,1)
ax5.hist(Stat_Response, bins = 16, label = 'Response', alpha = 0.8, color = 'red', edgecolor = 'red')
ax5.bar(S1x, S1y, color = 'blue', width = 15, alpha = 0.5)
ax5.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax5.tick_params(axis = 'x', labelsize = 8)
ax5.tick_params(axis = 'y', labelsize = 8)
ax5.set_xlabel("Angle (Degrees)", fontsize = 12)
ax5.set_ylabel("Number of Responses", fontsize = 12)
ax5.grid()
plt.title("Only Static Trial Responses", fontsize = 16)
plt.show()

fig6, ax6 = plt.subplots(1,1)
ax6.hist(Dyn_Response, bins = 16, label = 'Response', alpha = 0.8, color = 'red', edgecolor = 'red')
ax6.bar(D1x, D1y, color = 'blue', width = 15, alpha = 0.5)
ax6.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax6.tick_params(axis = 'x', labelsize = 8)
ax6.tick_params(axis = 'y', labelsize = 8)
ax6.set_xlabel("Angle (Degrees)", fontsize = 12)
ax6.set_ylabel("Number of Responses", fontsize = 12)
ax6.grid()
plt.title("Only Dynamic Trial Responses", fontsize = 16)
plt.show()
