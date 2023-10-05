import csv
import numpy as np
from numpy import genfromtxt
import math
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
#import needed libraries above

"""
#formulas for use in perception modeling
epsilon = |v-y|, v = desired response, y = user response
#linear model(xp and I are 0,1 scales and A is normal amplitude of intensity)
#xp=I1/(I1+I0)
#I1 = xp
#I0 = 1 - xp
#I = (A - A0)/(A1+A0)

#power model
#xp = I1^2/(I0^2 + I1^2)
#I0 = sqrt(1-xp)
#I1 = sqrt(xp)

#more than two tactors
#a = (vM)%N
#b=(a+1)%N
#xp = vM-a
#v is the mapping of 0-360, a and b are denoting tactor pairs for the needed phantom sensation, % is the modulo operator
"""

#file directory for data files
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
#all files for perfroming analysis
"""
filetypes = {
    1:"participant01_computed.csv",
    2:"participant03_computed.csv",
    3:"participant04_computed.csv",
    4:"participant05_computed.csv",
    5:"participant06_computed.csv",
    6:"participant07_computed.csv",
    7:"participant09_computed.csv",
    8:"participant08_computed.csv",
    9:"participant010_computed.csv",
    10:"participant011_computed.csv",
    11:"participant012_computed.csv",
    12:"participant013_computed.csv",
    13:"participant014_computed.csv",
    14:"participant015_computed.csv",
    15:"participant016_computed.csv",
    16:"participant017_computed.csv",
    17:"participant018_computed.csv",
    18:"participant019_computed.csv",
    19:"participant020_computed.csv",
    20:"participant021_computed.csv",
    21:"participant022_computed.csv",
    22:"participant023_computed.csv"}
"""
filetypes = {
    1:"participant018_computed.csv"}


#function for doing the angle calculation
#tactor_spot is used for computing the ratio for the location of the xp value according to the tactor pair in use
def Arduino_Mag(angle):
   sev = 0
   Motor1 = 0
   Motor2 = 0
   Motor3 = 0
   Motor4 = 0
   Motor5 = 0
   Motor6 = 0
   Motor7 = 0
   Motor8 = 0
   if angle < 0 or angle > 360:
      sev = 0
      Motor1 = 0
      Motor2 = 0
      Motor3 = 0
      Motor4 = 0
      Motor5 = 0
      Motor6 = 0
      Motor7 = 0
      Motor8 = 0
      tactor_spot = 1
      Motor_Data = [Motor1, Motor2, tactor_spot]

   elif angle <= 45:
      speed = (45 - angle)/45
      Motor1 = speed
      Motor2 = 1 - speed
      tactor_spot = 1
      Motor_Data = [Motor1, Motor2, tactor_spot]

   elif angle <= 90:
      speed = (90 - angle)/(90-45)
      Motor2 = speed
      Motor3 = 1 - speed
      tactor_spot = 2
      Motor_Data = [Motor2, Motor3, tactor_spot]

   elif angle <= 135:
      speed = (135 - angle)/(135 - 90)
      Motor3 = speed
      Motor4 = 1 - speed
      tactor_spot = 3
      Motor_Data = [Motor3, Motor4, tactor_spot]

   elif angle <= 180:
      speed = (180 - angle)/(180 - 135)
      Motor4 = speed
      Motor5 = 1 - speed
      tactor_spot = 4
      Motor_Data = [Motor4, Motor5, tactor_spot]

   elif angle <= 225:
      speed = (225 - angle)/(225 - 180)
      Motor5 = speed
      Motor6 = 1 - speed
      tactor_spot = 5
      Motor_Data = [Motor5, Motor6, tactor_spot]

   elif angle <= 270:
      speed = (270 - angle)/(270 - 225)
      Motor6 = speed
      Motor7 = 1 - speed
      tactor_spot = 6
      Motor_Data = [Motor6, Motor7, tactor_spot]

   elif angle <= 315:
      speed = (315 - angle)/(315 - 270)
      Motor7 = speed
      Motor8 = 1 - speed
      tactor_spot = 7
      Motor_Data = [Motor7, Motor8, tactor_spot]

   elif angle <= 360:
      speed = (360 - angle)/(360 - 315)
      Motor8 = speed
      Motor1 = 1 - speed
      tactor_spot = 8
      Motor_Data = [Motor8, Motor1, tactor_spot]
      
   return Motor_Data

#function for normalizing user response from 0-360 to 0-1
def value_normalizing(x):
   des_max = 360
   des_min = 0
   normalized = (x - des_min)/(des_max-des_min)
   return normalized

#value for mapping the motor amplitude intenities to 0-255
def value_mapping(x):
   amp_max = 360
   amp_min = 0
   norm_mag = (x - amp_min)/(amp_max-amp_min)
   N=8
   M=8
   v = x/360.0
   a = np.mod(np.floor(v*M),N)
   b = np.mod((1+a),N)
   xppp = v*M - a
   return (xppp*((a+1)/8))

#function for computing the within interval xp value
def Tactorp_within(x):
   N=8
   M=8
   v = x/360.0
   a = np.mod(np.floor(v*M),N)
   b = np.mod((1+a),N)
   xp = v*M - a
   return xp

#function computes the xp value based upon the xp from within-interval (not correct) 
def perception_model(x):
   I0l = 1 - x
   I1l = x
   I0p = np.sqrt(1-x)
   I1p = np.sqrt(x)
   xpl = I1l/(I0l+I1l)
   xpp = I1p**2/(I0p**2 + I1p**2)
   return [xpl, xpp]

#estimate xp based on motor intensity from Arduino_Mag function
def intensity_estimate(x):
   l = (x[1])/(x[0]+x[1])
   p = (x[1]**2)/((x[0]**2)+(x[1]**2))
   #creating xp with an attempt to scale based on which section its in (do not think this is ok)
   xp = [l*(x[2]/8), p*(x[2]/8)]
   return xp

#set only one participant data type for testing methodology
Data = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)


angles = [0, 15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345, 360]

#normalize the angles according to 0-360 scale
normal_angles = [value_mapping(item) for item in angles]
#get within tactor xp to plot models against
normal_angle = [Tactorp_within(item) for item in angles]

#normalizing angles from static and dynamic trials split into desired and response process using while loops
normal_angles_stat = []
normal_response_stat = []
normal_angles_dyn = []
normal_response_dyn = []
i = 0
while i <= len(Data)-1:
   if (Data[i][0] == 0):
      normal_angles_stat.append(value_normalizing(Data[i][1]))
      normal_response_stat.append(value_normalizing(Data[i][5]))
   elif(Data[i][0] == 1):
      normal_angles_dyn.append(value_normalizing(Data[i][1]))
      normal_response_dyn.append(value_normalizing(Data[i][5]))
   i = i + 1

#converting normalized angle data into combined lists for easy ploting
normal_data_stat =[normal_angles_stat, normal_response_stat]
normal_data_dyn = [normal_angles_dyn, normal_response_dyn]

#the code here is to try and make models scale nicely from 0-1 for all data (math does not work out)
#creating angle set to make nice model lines
angles_model = np.linspace(0,1,len(angles))

#find xp for linear and power using motor intensity
motor_xpl = []
motor_xpp = []
k = 0
while k<=len(angles)-1:
   val = Arduino_Mag(angles[k])
   est = intensity_estimate(val)
   xll = est[0]
   xpp = est[1]
   motor_xpl.append(xll)
   motor_xpp.append(xpp)
   k = k + 1

#multiple tactor math test for models
"""
N = 8. #number of tactors in the system
M = 8. #number of segments between tactors in system

test = 90./360.
a = np.mod(np.floor(test*M),N)
b = np.mod((1+a),N)
xp = (test*M) - a


I0l = 1 - xp
I1l = xp

I0p = np.sqrt(1-xp)
I1p = np.sqrt(xp)

xpl = I1l/(I0l+I1l)
xpp = I1p**2/(I0p**2+I1p**2)
print(test, a, b, xp, I0l, I1l, xpl, I0p, I1p, xpp)
"""
#finding model estimations using angles to get intensity and estimated xp (bad method)
"""
modela_data = [Tactorp_within(item) for item in angles]
modelr_data = [perception_model(item) for item in modela_data]
lin_modelr_data = []
po_modelr_data = []
j = 0
while j<= len(modelr_data) - 1:
   lin_modelr_data.append(modelr_data[j][0])
   po_modelr_data.append(modelr_data[j][1])
   j = j + 1
"""
#z_stat = np.polyfit(normal_data_stat[:][0],normal_data_stat[:][1], 1)
#p_stat = np.poly1d(z_stat)
fig1 = plt.figure("Figure 1")

plt.scatter(normal_data_stat[:][0],normal_data_stat[:][1])
plt.plot(normal_angles, motor_xpl, label = "Linear Model", color = "tab:red")
plt.plot(normal_angles, motor_xpp, label = "Power Model", color = "k")

plt.show()

fig2 = plt.figure("Figure 2")
#z_dyn = np.polyfit(normal_data_dyn[:][0],normal_data_dyn[:][1], 1)
#p_dyn = np.poly1d(z_dyn)
plt.scatter(normal_data_dyn[:][0],normal_data_dyn[:][1])
plt.plot(normal_angles, motor_xpl, label = "Linear Model", color = "tab:red")
plt.plot(angles_model, motor_xpp, label = "Power Model", color = "k")
plt.show()
