import csv
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#linux directory for the means and standard error data
#DIR = "/home/zal5/iiwa_ws/src/haptics_controls/src/Plots/"
#Mac directory
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
#DIR = "C:\\Users\\4paws\\Documents\\Research_Test_Stuff\\"
#names of the means and standard error data files
filetypes = {
	1:"d1means.csv",
	2:"d1ses.csv",
	3:"d2means.csv",
	4:"d2ses.csv",
	5:"d3means.csv",
	6:"d3ses.csv",
	7:"d4means.csv",
	8:"d4ses.csv",
	9:"d5means.csv",
	10:"d5ses.csv"}

#generate variables to load all the file data into in the float variable type
means1 = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
ses1 = genfromtxt(DIR+filetypes[2],delimiter=',',dtype=float)
means2 = genfromtxt(DIR+filetypes[3],delimiter=',',dtype=float)
ses2 = genfromtxt(DIR+filetypes[4],delimiter=',',dtype=float)
means3 = genfromtxt(DIR+filetypes[5],delimiter=',',dtype=float)
ses3 = genfromtxt(DIR+filetypes[6],delimiter=',',dtype=float)
means4 = genfromtxt(DIR+filetypes[7],delimiter=',',dtype=float)
ses4 = genfromtxt(DIR+filetypes[8],delimiter=',',dtype=float)
means5 = genfromtxt(DIR+filetypes[9],delimiter=',',dtype=float)
ses5 = genfromtxt(DIR+filetypes[10],delimiter=',',dtype=float)

#delete statements to remove the original angle lables and condition headers as they show up as nans
means1 = np.delete(means1,0,0)
means1 = np.delete(means1,0,1)

means2 = np.delete(means2,0,0)
means2 = np.delete(means2,0,1)

means3 = np.delete(means3,0,0)
means3 = np.delete(means3,0,1)

means4 = np.delete(means4,0,0)
means4 = np.delete(means4,0,1)

means5 = np.delete(means5,0,0)
means5 = np.delete(means5,0,1)

ses1 = np.delete(ses1,0,0)
ses1 = np.delete(ses1,0,1)

ses2 = np.delete(ses2,0,0)
ses2 = np.delete(ses2,0,1)

ses3 = np.delete(ses3,0,0)
ses3 = np.delete(ses3,0,1)

ses4 = np.delete(ses4,0,0)
ses4 = np.delete(ses4,0,1)

ses5 = np.delete(ses5,0,0)
ses5 = np.delete(ses5,0,1)

#new varible containing the angle labels
angle = [15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345]
#new variable containing the magnitude labels
mags = [0.25, 0.50, 0.75, 1.00]


#intialization of variables used in the while loops
stat_ang_means1 = []
dyn_ang_means1 = []
stat_ang_means2 = []
dyn_ang_means2 = []
stat_mag_means3 = []
dyn_mag_means3 = []

stat_ang_se1 = []
dyn_ang_se1 = []
stat_ang_se2 = []
dyn_ang_se2 = []
stat_mag_se3 = []
dyn_mag_se3 = []

mag_25_means4 = []
mag_50_means4 = []
mag_75_means4 = []
mag_10_means4 = []
mag_25_se4 = []
mag_50_se4 = []
mag_75_se4 = []
mag_10_se4 = []

ang_25_means5 = []
ang_50_means5 = []
ang_75_means5 = []
ang_10_means5 = []
ang_25_se5 = []
ang_50_se5 = []
ang_75_se5 = []
ang_10_se5 = []
#while loops to seperate the means ans standard errors into seperate variables based on condition
i = 0
j = 0
k = 0
l = 0
while i <= len(means1) - 1:
	stat_ang_means1.append(means1[i][0])
	dyn_ang_means1.append(means1[i][1])
	stat_ang_means2.append(means2[i][0])
	dyn_ang_means2.append(means2[i][1])
	
	stat_ang_se1.append(ses1[i][0])
	dyn_ang_se1.append(ses1[i][1])
	stat_ang_se2.append(ses2[i][0])
	dyn_ang_se2.append(ses2[i][1])
	
	i = i + 1
	
while j <= len(means3) - 1:
	stat_mag_means3.append(means3[j][0])
	dyn_mag_means3.append(means3[j][1])
	stat_mag_se3.append(ses3[j][0])
	dyn_mag_se3.append(ses3[j][1])
	
	j = j + 1

while k <= len(means5) - 1:
	ang_25_means5.append(means5[k][0])
	ang_50_means5.append(means5[k][1])
	ang_75_means5.append(means5[k][2])
	ang_10_means5.append(means5[k][3])
	ang_25_se5.append(ses5[k][0])
	ang_50_se5.append(ses5[k][1])
	ang_75_se5.append(ses5[k][2])
	ang_10_se5.append(ses5[k][3])
	k = k + 1
	
while l <= len(means4) - 1:
	mag_25_means4.append(means4[l][0])
	mag_50_means4.append(means4[l][1])
	mag_75_means4.append(means4[l][2])
	mag_10_means4.append(means4[l][3])
	mag_25_se4.append(ses4[l][0])
	mag_50_se4.append(ses4[l][0])
	mag_75_se4.append(ses4[l][0])
	mag_10_se4.append(ses4[l][0])
	l = l + 1
	
#obtain error for each quadrant of the hand
quad1means = [stat_ang_means1[0], stat_ang_means1[1], stat_ang_means1[2], stat_ang_means1[3]]
quad2means = [stat_ang_means1[4], stat_ang_means1[5], stat_ang_means1[6], stat_ang_means1[7]]
quad3means = [stat_ang_means1[8], stat_ang_means1[9], stat_ang_means1[10], stat_ang_means1[11]]
quad4means = [stat_ang_means1[12], stat_ang_means1[13], stat_ang_means1[14], stat_ang_means1[15]]

quad1ses = [stat_ang_se1[0], stat_ang_se1[1], stat_ang_se1[2], stat_ang_se1[3]]
quad2ses = [stat_ang_se1[4], stat_ang_se1[5], stat_ang_se1[6], stat_ang_se1[7]]
quad3ses = [stat_ang_se1[8], stat_ang_se1[9], stat_ang_se1[10], stat_ang_se1[11]]
quad4ses = [stat_ang_se1[12], stat_ang_se1[13], stat_ang_se1[14], stat_ang_se1[15]]

meanquad1 = sum(quad1means)/len(quad1means)
meanquad2 = sum(quad2means)/len(quad2means)
meanquad3 = sum(quad3means)/len(quad3means)
meanquad4 = sum(quad4means)/len(quad4means)

seavg1 = sum(quad1ses)/len(quad1ses)
seavg2 = sum(quad2ses)/len(quad2ses)
seavg3 = sum(quad3ses)/len(quad3ses)
seavg4 = sum(quad4ses)/len(quad4ses)

stat_quad_means = [meanquad1, meanquad2, meanquad3, meanquad4]
stat_quad_se = [seavg1, seavg2, seavg3, seavg4]

dquad1means = [dyn_ang_means1[0], dyn_ang_means1[1], dyn_ang_means1[2], dyn_ang_means1[3]]
dquad2means = [dyn_ang_means1[4], dyn_ang_means1[5], dyn_ang_means1[6], dyn_ang_means1[7]]
dquad3means = [dyn_ang_means1[8], dyn_ang_means1[9], dyn_ang_means1[10], dyn_ang_means1[11]]
dquad4means = [dyn_ang_means1[12], dyn_ang_means1[13], dyn_ang_means1[14], dyn_ang_means1[15]]

dquad1ses = [dyn_ang_se1[0], dyn_ang_se1[1], dyn_ang_se1[2], dyn_ang_se1[3]]
dquad2ses = [dyn_ang_se1[4], dyn_ang_se1[5], dyn_ang_se1[6], dyn_ang_se1[7]]
dquad3ses = [dyn_ang_se1[8], dyn_ang_se1[9], dyn_ang_se1[10], dyn_ang_se1[11]]
dquad4ses = [dyn_ang_se1[12], dyn_ang_se1[13], dyn_ang_se1[14], dyn_ang_se1[15]]

dmeanquad1 = sum(dquad1means)/len(dquad1means)
dmeanquad2 = sum(dquad2means)/len(dquad2means)
dmeanquad3 = sum(dquad3means)/len(dquad3means)
dmeanquad4 = sum(dquad4means)/len(dquad4means)

dseavg1 = sum(dquad1ses)/len(dquad1ses)
dseavg2 = sum(dquad2ses)/len(dquad2ses)
dseavg3 = sum(dquad3ses)/len(dquad3ses)
dseavg4 = sum(dquad4ses)/len(dquad4ses)

dyn_quad_means = [dmeanquad1, dmeanquad2, dmeanquad3, dmeanquad4]
dyn_quad_se = [dseavg1, dseavg2, dseavg3, dseavg4]

"""
#non-grouped plot for interation of angle error and commanded angle
fig1 = plt.figure("Figure 1")
plt.plot(angle,stat_ang_means1, label = "Static Start", linewidth = 2.0, color = 'r')
plt.errorbar(angle,stat_ang_means1, yerr = stat_ang_se1, fmt='o', capsize = 4, marker = '', ecolor = 'tab:red')

plt.plot(angle,dyn_ang_means1, label = "Dynamic Start", linewidth=2.0, color='b')
plt.errorbar(angle,dyn_ang_means1, yerr = dyn_ang_se1, fmt='o', capsize = 4, marker = '', ecolor = 'tab:blue')
plt.xlabel("Commanded Angle", fontsize = 35)
plt.ylabel("Error From Commanded Angle (degrees)", fontsize = 35, wrap = True)
plt.legend(loc = 'best', title = 'Movement Condition', title_fontsize = 35, fontsize = 35)
plt.yticks(fontsize = 35)
plt.xticks(fontsize = 35)
plt.axis([None, None, 20, 160])
plt.show()
"""
"""
#non-grouped plot for interation of mag error and commanded angle 
fig2 = plt.figure("Figure 2")
plt.plot(angle,stat_ang_means2, label = "Static Start", linewidth = 2.0, color = 'r')
plt.errorbar(angle,stat_ang_means2, yerr = stat_ang_se2, fmt='o', capsize = 4, marker = '', ecolor = 'tab:red')

plt.plot(angle,dyn_ang_means2, label = "Dynamic Start", linewidth=2.0, color='b')
plt.errorbar(angle,dyn_ang_means2, yerr = dyn_ang_se2, fmt = 'o', capsize = 4, marker = '', ecolor = 'tab:blue')

plt.xlabel("Commaned Angle", fontsize = 14)
plt.ylabel("Error From Commanded Magnitude (Percent of Max Speed)", fontsize = 14)
plt.legend(loc = 'best', title = 'Movement Condition', fontsize = 14)
plt.show()
"""

fig4 = plt.figure("Figure 4")
X_axis = np.arange(len(mags))
	
plt.bar(X_axis - 0.2,stat_mag_means3, 0.4, label = "Static Start", color = 'r')
plt.errorbar(X_axis - 0.2,stat_mag_means3, yerr = stat_mag_se3, fmt = 'o', linewidth = 6.0, capsize = 15, marker = '', ecolor = 'k')

plt.bar(X_axis + 0.2,dyn_mag_means3, 0.4, label = "Dynamic Start", color = 'b')
plt.errorbar(X_axis + 0.2,dyn_mag_means3, yerr = dyn_mag_se3, fmt = 'o', linewidth = 6.0, capsize = 15, marker = '', ecolor = 'k')

plt.yticks(fontsize = 35)
plt.xticks(X_axis,mags, fontsize = 30)
plt.xlabel("Target Speed", fontsize = 30)
plt.ylabel("Speed Error (m/s)", fontsize = 30, wrap = True)
plt.legend(loc = 'best', title = 'Movement Condition', title_fontsize = 30, fontsize = 30)
plt.axis([None, None, 0, 0.10])

plt.show()


Hand_Quadrants = ["Ulnar-Dorsal", "Radial-Dorsal", "Radial-Ventral", "Ulnar-Ventral"]
fig5 = plt.figure("Figure 5")
xaxis = np.arange(len(Hand_Quadrants))

plt.bar(xaxis - 0.2,stat_quad_means, 0.4, label = "Static Start", color = 'r')
plt.errorbar(xaxis - 0.2,stat_quad_means, yerr = stat_quad_se, fmt = 'o', linewidth = 6.0, capsize = 15, marker = '', ecolor = 'k')

plt.bar(xaxis + 0.2,dyn_quad_means, 0.4, label = "Dynamic Start", color='b')
plt.errorbar(xaxis + 0.2,dyn_quad_means, yerr = dyn_quad_se, fmt = 'o', linewidth = 6.0, capsize = 15, marker = '', ecolor = 'k')

plt.yticks(fontsize = 35)
plt.xticks(xaxis, Hand_Quadrants, fontsize = 20, wrap = True)
plt.xlabel("Forearm Location", fontsize = 30, wrap = True)
plt.ylabel("Angle Error (degrees)", fontsize = 30, wrap = True)
plt.legend(loc = 'best', title = 'Movement Condition', title_fontsize = 30, fontsize = 30)

plt.axis([None, None, -45, 5])

#plt.savefig("Quadrant_Angle_Error.png", bbox_inches = "tight")
plt.show()


#group speed (error) means and standard deviations according to intensity of cue and angular direction
mag_quad_means1_25 = [mag_25_means4[0], mag_25_means4[1], mag_25_means4[2], mag_25_means4[3]]
mag_quad_means2_25 = [mag_25_means4[4], mag_25_means4[5], mag_25_means4[6], mag_25_means4[7]]
mag_quad_means3_25 = [mag_25_means4[8], mag_25_means4[9], mag_25_means4[10], mag_25_means4[11]]
mag_quad_means4_25 = [mag_25_means4[12], mag_25_means4[13], mag_25_means4[14], mag_25_means4[15]]

mag_quad_means1_50 = [mag_50_means4[0], mag_50_means4[1], mag_50_means4[2], mag_50_means4[3]]
mag_quad_means2_50 = [mag_50_means4[4], mag_50_means4[5], mag_50_means4[6], mag_50_means4[7]]
mag_quad_means3_50 = [mag_50_means4[8], mag_50_means4[9], mag_50_means4[10], mag_50_means4[11]]
mag_quad_means4_50 = [mag_50_means4[12], mag_50_means4[13], mag_50_means4[14], mag_50_means4[15]]

mag_quad_means1_75 = [mag_75_means4[0], mag_75_means4[1], mag_75_means4[2], mag_75_means4[3]]
mag_quad_means2_75 = [mag_75_means4[4], mag_75_means4[5], mag_75_means4[6], mag_75_means4[7]]
mag_quad_means3_75 = [mag_75_means4[8], mag_75_means4[9], mag_75_means4[10], mag_75_means4[11]]
mag_quad_means4_75 = [mag_75_means4[12], mag_75_means4[13], mag_75_means4[14], mag_75_means4[15]]

mag_quad_means1_10 = [mag_10_means4[0], mag_10_means4[1], mag_10_means4[2], mag_10_means4[3]]
mag_quad_means2_10 = [mag_10_means4[4], mag_10_means4[5], mag_10_means4[6], mag_10_means4[7]]
mag_quad_means3_10 = [mag_10_means4[8], mag_10_means4[9], mag_10_means4[10], mag_10_means4[11]]
mag_quad_means4_10 = [mag_10_means4[12], mag_10_means4[13], mag_10_means4[14], mag_10_means4[15]]


#print(sum(mag_quad_means1_25)/4)
p1 = sum(mag_quad_means1_25)/4
p2 = sum(mag_quad_means2_25)/4
p3 = sum(mag_quad_means3_25)/4
p4 = sum(mag_quad_means4_25)/4

a = [p1, p2, p3, p4]

q1 = sum(mag_quad_means1_50)/4
q2 = sum(mag_quad_means2_50)/4
q3 = sum(mag_quad_means3_50)/4
q4 = sum(mag_quad_means4_50)/4

b = [q1, q2, q3, q4]
w1 = sum(mag_quad_means1_75)/4
w2 = sum(mag_quad_means2_75)/4
w3 = sum(mag_quad_means3_75)/4
w4 = sum(mag_quad_means4_75)/4

c = [w1, w2, w3, w4]

e1 = sum(mag_quad_means1_10)/4
e2 = sum(mag_quad_means2_10)/4
e3 = sum(mag_quad_means3_10)/4
e4 = sum(mag_quad_means4_10)/4

d = [e1, e2, e3, e4]

mag_quad_se1_25 = [mag_25_se4[0], mag_25_se4[1], mag_25_se4[2], mag_25_se4[3]]
mag_quad_se2_25 = [mag_25_se4[4], mag_25_se4[5], mag_25_se4[6], mag_25_se4[7]]
mag_quad_se3_25 = [mag_25_se4[8], mag_25_se4[9], mag_25_se4[10], mag_25_se4[11]]
mag_quad_se4_25 = [mag_25_se4[12], mag_25_se4[13], mag_25_se4[14], mag_25_se4[15]]

mag_quad_se1_50 = [mag_50_se4[0], mag_50_se4[1], mag_50_se4[2], mag_50_se4[3]]
mag_quad_se2_50 = [mag_50_se4[4], mag_50_se4[5], mag_50_se4[6], mag_50_se4[7]]
mag_quad_se3_50 = [mag_50_se4[8], mag_50_se4[9], mag_50_se4[10], mag_50_se4[11]]
mag_quad_se4_50 = [mag_50_se4[12], mag_50_se4[13], mag_50_se4[14], mag_50_se4[15]]

mag_quad_se1_75 = [mag_75_se4[0], mag_75_se4[1], mag_75_se4[2], mag_75_se4[3]]
mag_quad_se2_75 = [mag_75_se4[4], mag_75_se4[5], mag_75_se4[6], mag_75_se4[7]]
mag_quad_se3_75 = [mag_75_se4[8], mag_75_se4[9], mag_75_se4[10], mag_75_se4[11]]
mag_quad_se4_75 = [mag_75_se4[12], mag_75_se4[13], mag_75_se4[14], mag_75_se4[15]]

mag_quad_se1_10 = [mag_10_se4[0], mag_10_se4[1], mag_10_se4[2], mag_10_se4[3]]
mag_quad_se2_10 = [mag_10_se4[4], mag_10_se4[5], mag_10_se4[6], mag_10_se4[7]]
mag_quad_se3_10 = [mag_10_se4[8], mag_10_se4[9], mag_10_se4[10], mag_10_se4[11]]
mag_quad_se4_10 = [mag_10_se4[12], mag_10_se4[13], mag_10_se4[14], mag_10_se4[15]]

a1 = [sum(mag_quad_se1_25)/4, sum(mag_quad_se2_25)/4, sum(mag_quad_se3_25)/4, sum(mag_quad_se4_25)/4]
b1 = [sum(mag_quad_se1_50)/4, sum(mag_quad_se2_50)/4, sum(mag_quad_se3_50)/4, sum(mag_quad_se4_50)/4]
c1 = [sum(mag_quad_se1_75)/4, sum(mag_quad_se2_75)/4, sum(mag_quad_se3_75)/4, sum(mag_quad_se4_75)/4]
d1 = [sum(mag_quad_se1_10)/4, sum(mag_quad_se2_10)/4, sum(mag_quad_se3_10)/4, sum(mag_quad_se4_10)/4]


fig7 = plt.figure("Figure 7")
barwidth = 0.2
xaxt = np.arange(len(Hand_Quadrants))
k1 = [x - barwidth for x in xaxt]
k2 = xaxt
k3 = [x + barwidth for x in xaxt]
k4 = [x + 2*barwidth for x in xaxt]

plt.bar(k1, a, barwidth, color = 'r')

plt.bar(k2, b, barwidth, color = 'b')

plt.bar(k3, c, barwidth, color = 'g')

plt.bar(k4, d, barwidth, color = 'tab:purple')

plt.legend(labels = mags, loc = 'best', title = 'Target Speed', title_fontsize = 25, fontsize = 25)

plt.errorbar(k1, a, yerr = a1, fmt = 'o', linewidth = 2.0, capsize = 5, marker = '', ecolor = 'k')
plt.errorbar(k2, b, yerr = b1, fmt = 'o', linewidth = 2.0, capsize = 5, marker = '', ecolor = 'k')
plt.errorbar(k3, c, yerr = c1, fmt = 'o', linewidth = 2.0, capsize = 5, marker = '', ecolor = 'k')
plt.errorbar(k4, d, yerr = d1, fmt = 'o', linewidth = 2.0, capsize = 5, marker = '', ecolor = 'k')

plt.xticks(xaxt, Hand_Quadrants, fontsize = 30)
plt.yticks(fontsize = 30)
plt.ylabel("Speed Error (m/s)", fontsize = 30)
plt.xlabel("Target Angle (degrees)", fontsize = 30)

plt.show()
