import csv
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.metrics import mean_squared_error
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
#math for testing modulus
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

#function for doing the angle calculation
#tactor_spot is used for computing the ratio for the location of the xp value according to the tactor pair in use
def Arduino_Magl(angle):
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
      tactor_spot = 0
      Motor_Data = [Motor1, Motor2, tactor_spot]

   elif angle <= 45:
      speed = (45 - angle)/45
      Motor1 = speed
      Motor2 = 1 - speed
      tactor_spot = 0
      Motor_Data = [Motor1, Motor2, tactor_spot]

   elif angle <= 90:
      speed = (90 - angle)/(90-45)
      Motor2 = speed
      Motor3 = 1 - speed
      tactor_spot = 1
      Motor_Data = [Motor2, Motor3, tactor_spot]

   elif angle <= 135:
      speed = (135 - angle)/(135 - 90)
      Motor3 = speed
      Motor4 = 1 - speed
      tactor_spot = 2
      Motor_Data = [Motor3, Motor4, tactor_spot]

   elif angle <= 180:
      speed = (180 - angle)/(180 - 135)
      Motor4 = speed
      Motor5 = 1 - speed
      tactor_spot = 3
      Motor_Data = [Motor4, Motor5, tactor_spot]

   elif angle <= 225:
      speed = (225 - angle)/(225 - 180)
      Motor5 = speed
      Motor6 = 1 - speed
      tactor_spot = 4
      Motor_Data = [Motor5, Motor6, tactor_spot]

   elif angle <= 270:
      speed = (270 - angle)/(270 - 225)
      Motor6 = speed
      Motor7 = 1 - speed
      tactor_spot = 5
      Motor_Data = [Motor6, Motor7, tactor_spot]

   elif angle <= 315:
      speed = (315 - angle)/(315 - 270)
      Motor7 = speed
      Motor8 = 1 - speed
      tactor_spot = 6
      Motor_Data = [Motor7, Motor8, tactor_spot]

   elif angle <= 360:
      speed = (360 - angle)/(360 - 315)
      Motor8 = speed
      Motor1 = 1 - speed
      tactor_spot = 7
      Motor_Data = [Motor8, Motor1, tactor_spot]
      
   return Motor_Data

#function for normalizing user response from 0-360 to 0-1
def value_normalizing(x):
   des_max = 360
   des_min = 0
   normalized = (x - des_min)/(des_max-des_min)
   return normalized

#estimate xp based on motor intensity from Arduino_Mag function
def intensity_estimate(x):
   #within interval linear model estimation
   lw = (x[1])/(x[0]+x[1])
   #within interval power model estimation
   pw = (x[1]**2)/((x[0]**2)+(x[1]**2))
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match x[2]:
      case 0:
         smin = 0
         smax = 0.125
      case 1:
         smin = 0.125
         smax = 0.250
      case 2:
         smin = 0.250
         smax = 0.375
      case 3:
         smin = 0.375
         smax = 0.500
      case 4:
         smin = 0.500
         smax = 0.625
      case 5:
         smin = 0.625
         smax = 0.750
      case 6:
         smin = 0.750
         smax = 0.875
      case 7:
         smin = 0.875
         smax = 1.000
   #compute the scaled value of the linear and power model estimation
   l = (lw * (smax-smin)) + smin
   p = (pw * (smax-smin)) + smin
   #combine into a single varible for easier use
   xp = [l, p]
   return xp


def intensity_estimate_adjusted(x,S0,S1):
   #within interval linear model estimation
   lw = (x[1]*S1[0])/(x[0]*S0[0]+x[1]*S1[0])
   #within interval power model estimation
   pw = (S1[1]*x[1]**2)/((S0[1]*x[0]**2)+(S1[1]*x[1]**2))
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match x[2]:
      case 0:
         smin = 0
         smax = 0.125
      case 1:
         smin = 0.125
         smax = 0.250
      case 2:
         smin = 0.250
         smax = 0.375
      case 3:
         smin = 0.375
         smax = 0.500
      case 4:
         smin = 0.500
         smax = 0.625
      case 5:
         smin = 0.625
         smax = 0.750
      case 6:
         smin = 0.750
         smax = 0.875
      case 7:
         smin = 0.875
         smax = 1.000
   #compute the scaled value of the linear and power model estimation
   l = (lw * (smax-smin)) + smin
   p = (pw * (smax-smin)) + smin
   #combine into a single varible for easier use
   xp = [l, p]
   return xp


def linear_estimate(S0, S1, data):
   #within interval linear model estimation
   lw = (S1*data[1])/(S0*data[0]+S1*data[1])
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match data[2]:
      case 0:
         smin = 0
         smax = 0.125
      case 1:
         smin = 0.125
         smax = 0.250
      case 2:
         smin = 0.250
         smax = 0.375
      case 3:
         smin = 0.375
         smax = 0.500
      case 4:
         smin = 0.500
         smax = 0.625
      case 5:
         smin = 0.625
         smax = 0.750
      case 6:
         smin = 0.750
         smax = 0.875
      case 7:
         smin = 0.875
         smax = 1.000
   #compute the scaled value of the linear and power model estimation
   l = (lw * (smax-smin)) + smin
   return l

def power_estimate(S0, S1, data):
   #within interval linear model estimation
   pw = (S1*(data[1]**2))/(S0*(data[0]**2)+S1*(data[1]**2))
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match data[2]:
      case 0:
         smin = 0
         smax = 0.125
      case 1:
         smin = 0.125
         smax = 0.250
      case 2:
         smin = 0.250
         smax = 0.375
      case 3:
         smin = 0.375
         smax = 0.500
      case 4:
         smin = 0.500
         smax = 0.625
      case 5:
         smin = 0.625
         smax = 0.750
      case 6:
         smin = 0.750
         smax = 0.875
      case 7:
         smin = 0.875
         smax = 1.000
   #compute the scaled value of the linear and power model estimation
   p = (pw * (smax-smin)) + smin
   return p

#optimization linear function
def linear_optimum(x, angle, response):
   #get parameters
   S0 = x[0]
   S1 = x[1]
   predicted_response = []
   error = []
   error_sq = 0
   j = 0
   for i in angle:
      Motor = Arduino_Magl(i)
      predicted_response.append(power_estimate(S0, S1, Motor))
   
   while j <= len(angle)-1:
      err = response[j] - predicted_response[j]
      if (err > 0.5 or err < -0.5):
         if (angle[j] < 180):
            err = 1 + response[j] - predicted_response[j]
            error.append(err**2)
         elif(angle[j] > 180):
            err = 1 - response[j] - predicted_response[j]
            error.append(err**2)
      else:
         error.append(err**2)
      j = j+1
         
   error_sq = sum(error)
   
   return error_sq

def power_optimum(x, angle, response):
   #get parameters
   S0 = x[0]
   S1 = x[1]
   error_sq = 0
   j = 0
   error = []
   predicted_response = []
   for i in angle:
      Motor = Arduino_Magl(i)
      predicted_response.append(power_estimate(S0, S1, Motor))

   while j <= len(angle)-1:
      err = response[j] - predicted_response[j]
      if (err > 0.5 or err < -0.5):
         if (angle[j] < 180):
            err = 1 + response[j] - predicted_response[j]
            error.append(err**2)
         elif(angle[j] > 180):
            err = 1 - response[j] - predicted_response[j]
            error.append(err**2)
      else:
         error.append(err**2)
      j = j+1
         
   error_sq = sum(error)
   
   return error_sq
#file directory for data files
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
#all filetypes

filetypes = {
    1:"alldata.csv"}

#filetypes = {
#   1:"participant018_computed.csv"}

#set only one participant data type for testing methodology
Data = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
Data = np.delete(Data,0,0)
Data = np.delete(Data,0,1)
#angles used in experiment including 0 and 360 for completing model lines
#angles = [15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345]

#normalize the angles according to 0-360 scale
#normal_angles = [value_normalizing(item) for item in angles]

#creating angle set to make nice model lines
#angles_model = np.linspace(0,360,360)
#normalizing the model angle set
#angles_modeln = [value_normalizing(item) for item in angles_model]

#normalizing angles from static and dynamic trials split into desired and response process using while loops
normal_angles_stat = []
normal_response_stat = []
normal_angles_dyn = []
normal_response_dyn = []
angles_stat = []
angles_dyn = []
angles_statr = []
angles_dynr = []
i = 0
while i <= len(Data)-1:
   if (Data[i][0] == 0):
      normal_angles_stat.append(value_normalizing(Data[i][1]))
      normal_response_stat.append(value_normalizing(Data[i][5]))
      angles_stat.append(Data[i][1])
      angles_statr.append(Data[i][5])
   elif(Data[i][0] == 1):
      normal_angles_dyn.append(value_normalizing(Data[i][1]))
      normal_response_dyn.append(value_normalizing(Data[i][5]))
      angles_dyn.append(Data[i][1])
      angles_dynr.append(Data[i][5])
   i = i + 1
stat = [angles_stat, angles_statr]
#perform minimizing
guess = [5, 5]
#L-BFGS-B
#BFGS
#Nelder-Mead
#TNC
#SLSQP

bds = ((1, 50), (1, 50))
resultsl = optimize.minimize(linear_optimum, guess, method='Nelder-Mead', args = (angles_stat, normal_response_stat), bounds = bds)

resultsp = optimize.minimize(power_optimum, guess, method='Nelder-Mead', args = (angles_stat, normal_response_stat), bounds = bds)

resultdl = optimize.minimize(linear_optimum, guess, method='Nelder-Mead', args = (angles_dyn, normal_response_dyn), bounds = bds)

resultdp = optimize.minimize(power_optimum, guess, method='Nelder-Mead', args = (angles_dyn, normal_response_dyn), bounds = bds)

Ssl = resultsl.x
Ssp = resultsp.x
Sdl = resultdl.x
Sdp = resultdp.x

Ss0 = [Ssl[0], Ssp[0]]
Sd0 = [Sdl[0], Sdp[0]]
Ss1 = [Ssl[1], Ssp[1]]
Sd1 = [Sdl[1], Sdp[1]]


#converting normalized angle data into combined lists for easy ploting
#normal_data_stat =[normal_angles_stat, normal_response_stat, angles_stat, angles_statr]
#normal_data_dyn = [normal_angles_dyn, normal_response_dyn, angles_dyn, angles_dynr]

#find xp for linear and power using motor intensity
#motor_xpl = []
#motor_xpp = []
#motor_xplm = []
#motor_xppm = []
"""
k = 0
while k<=len(angles)-1:
   val = Arduino_Mag(angles[k])
   est = intensity_estimate(val)
   xll = est[0]
   xpp = est[1]
   motor_xpl.append(xll)
   motor_xpp.append(xpp)
   k = k + 1
"""
"""
nq = 0
while nq<=len(angles_model)-1:
   val = Arduino_Mag(angles_model[nq])
   est = intensity_estimate(val)
   xll = est[0]
   xpp = est[1]
   motor_xplm.append(xll)
   motor_xppm.append(xpp)
   nq = nq + 1
"""


#compute decoding error for against the models using while loop
decode = 0
epsilonsl = []
epsilonsp = []
epsilondl = []
epsilondp = []
while decode <= len(normal_response_stat)-1:
   #computing static error
   valsd = Arduino_Magl(angles_stat[decode])
   estsd = intensity_estimate_adjusted(valsd, Ss0, Ss1)
   xllsd = estsd[0]
   xppsd = estsd[1]
   errsl = normal_response_stat[decode] - xllsd
   errsp = normal_response_stat[decode] - xppsd

   #wrapping of error on 0-1 scale for static linear
   if (errsl > 0.5 or errsl < -0.5):
      if (angles_stat[decode] < 180):
         errorsl = 1 + normal_response_stat[decode] - xllsd
         epsilonsl.append(np.absolute(errorsl))
      elif(angles_stat[decode] > 180):
         errorsl = 1 - normal_response_stat[decode] - xllsd
         epsilonsl.append(np.absolute(errorsl))
   else:
      epsilonsl.append(np.absolute(errsl))
   #wrapping error on 0-1 scale for static power
   if (errsp > 0.5 or errsp < -0.5):
      if (angles_stat[decode] < 180):
         errorsp = 1 + normal_response_stat[decode] - xppsd
         epsilonsp.append(np.absolute(errorsp))
      elif(angles_stat[decode] > 180):
         errorsp = 1 - normal_response_stat[decode] - xppsd
         epsilonsp.append(np.absolute(errorsp))
   else:
      epsilonsp.append(np.absolute(errsp))
   decode = decode+1

decode2 = 0
while decode2 <= len(normal_response_dyn)-1:
   #computing dynamic error
   valdd = Arduino_Magl(angles_dyn[decode2])
   estdd = intensity_estimate_adjusted(valdd, Sd0, Sd1)
   xlldd = estdd[0]
   xppdd = estdd[1]
   errdl = normal_response_dyn[decode2] - xlldd
   errdp = normal_response_dyn[decode2] - xppdd
   #wrapping of error on 0-1 scale for dynamic linear
   if (errdl > 0.5 or errdl < -0.5):
      if (angles_dyn[decode2] < 180):
         errordl = 1 + normal_response_dyn[decode2] - xlldd
         epsilondl.append(np.absolute(errordl))
      elif(angles_dyn[decode2] > 180):
         errordl = 1 - normal_response_dyn[decode2] - xlldd
         epsilondl.append(np.absolute(errordl))
   else:
      epsilondl.append(np.absolute(errdl))

   #wrapping error on 0-1 scale for dynamic power
   if (errdp > 0.5 or errdp < -0.5):
      if (angles_dyn[decode2] < 180):
         errordp = 1 + normal_response_dyn[decode2] - xppdd
         epsilondp.append(np.absolute(errordp))
      elif(angles_dyn[decode2] > 180):
         errordp = 1 - normal_response_dyn[decode2] - xppdd
         epsilondp.append(np.absolute(errordp))
   else:
      epsilondp.append(np.absolute(errdp))
   decode2 = decode2+1

meansl = np.mean(epsilonsl)
meansp = np.mean(epsilonsp)
meandl = np.mean(epsilondl)
meandp = np.mean(epsilondp)

stdsl = np.std(epsilonsl)
stdsp = np.std(epsilonsp)
stddl = np.std(epsilondl)
stddp = np.std(epsilondp)
print("L Mean","L STD", "P Mean","P STD")
print("\n")
print(meansl, stdsl ,meansp, stdsp)
print("\n")
print(meandl, stddl ,meandp, stddp)
print("\n")
print("Variances")
varsl = np.var(epsilonsl)
varsp = np.var(epsilonsp)
vardl = np.var(epsilondl)
vardp = np.var(epsilondp)
print(varsl, varsp ,vardl, vardp)
print("\n")
stlp = stats.ttest_ind(a=epsilonsl, b=epsilonsp, equal_var=False)
dtlp = stats.ttest_ind(a=epsilondl, b=epsilondp, equal_var=False)
sldl = stats.ttest_ind(a=epsilonsl, b=epsilondl, equal_var=False)
spdp = stats.ttest_ind(a=epsilonsp, b=epsilondp, equal_var=False)
print("Statistic Results")
print(stlp)
print("\n")
print(dtlp)
print("\n")
print(sldl)
print("\n")
print(spdp)

##
##meansl = np.mean(epsilonsl)
##meansp = np.mean(epsilonsp)
##meandl = np.mean(epsilondl)
##meandp = np.mean(epsilondp)
##
##medsl = np.median(epsilonsl)
##medsp = np.median(epsilonsp)
##meddl = np.median(epsilondl)
##meddp = np.median(epsilondp)
##
##stdsl = np.std(epsilonsl)
##stdsp = np.std(epsilonsp)
##stddl = np.std(epsilondl)
##stddp = np.std(epsilondp)
##
##varsl = np.var(epsilonsl)
##varsp = np.var(epsilonsp)
##vardl = np.var(epsilondl)
##vardp = np.var(epsilondp)
##
##stlp = stats.ttest_ind(a=epsilonsl, b=epsilonsp, equal_var=True)
##dtlp = stats.ttest_ind(a=epsilondl, b=epsilondp, equal_var=True)
##sldl = stats.ttest_ind(a=epsilonsl, b=epsilondl, equal_var=True)
##spdp = stats.ttest_ind(a=epsilonsp, b=epsilondp, equal_var=True)
##print("L Mean","L STD", "P Mean","P STD")
##print("\n")
##print(meansl, stdsl ,meansp, stdsp)
##print("\n")
##print(meandl, stddl ,meandp, stddp)
##print("\n")
##print("Medians")
##print(medsl, medsp, meddl, meddp)
##print("\n")
###print("\n")
##print(stlp)
##print("\n")
##print(dtlp)
##print("\n")
##print(sldl)
##print("\n")
##print(spdp)
###print statements for testing
###print(angles)
###print("\n")
###print(normal_angles)
###print("\n")
###print(motor_xpl)
###print("\n")
###print(motor_xpp)
##
###plotting data and perception models
###plot of static data using trend lines of only experiment angles
##
##fig1 = plt.figure("Figure 1")
##plt.scatter(normal_data_stat[:][0],normal_data_stat[:][1], s=10)
##plt.plot(normal_angles, motor_xpl, label = "Linear Model", color = "tab:red", linewidth = 2)
##plt.plot(normal_angles, motor_xpp, label = "Power Model", color = "k", linewidth = 2)
##
###plot of static data using trendline with modeled angles
##fig2 = plt.figure("Figure 2")
##plt.scatter(normal_data_stat[:][0],normal_data_stat[:][1], s=15)
##plt.plot(angles_modeln, motor_xplm, label = "Linear Model", color = "tab:red", linewidth = 2)
##plt.plot(angles_modeln, motor_xppm, label = "Power Model", color = "k", linewidth = 2)
##plt.show()
##
###plot of dynamic data using trend lines of only experiment angles
##fig3 = plt.figure("Figure 3")
##plt.scatter(normal_data_dyn[:][0],normal_data_dyn[:][1], s=10)
##plt.plot(normal_angles, motor_xpl, label = "Linear Model", color = "tab:red", linewidth = 2)
##plt.plot(normal_angles, motor_xpp, label = "Power Model", color = "k", linewidth = 2)
##plt.show()
##
###plot of dynamic data using trendline with modeled angles
##fig4 = plt.figure("Figure 4")
##plt.scatter(normal_data_dyn[:][0],normal_data_dyn[:][1], s=15)
##plt.plot(angles_modeln, motor_xplm, label = "Linear Model", color = "tab:red", linewidth = 2)
##plt.plot(angles_modeln, motor_xppm, label = "Power Model", color = "k", linewidth = 2)
##plt.show()
