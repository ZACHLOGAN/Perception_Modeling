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
#function for pooled standard deviation
def pool_std(x,y):
   n1 = len(x)
   n2 = len(y)
   s1 = np.var(x)
   s2 = np.var(y)
   ss1 = np.std(x)
   ss2 = np.std(y)
   pooled_standard_deviation = np.sqrt((s1 + s2)/2)
   #pooled_standard_deviation = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
   return pooled_standard_deviation

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


def intensity_estimate_adjusted(x, SL, SP):
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match x[2]:
      case 0:
         smin = 0
         smax = 0.125
         #within interval linear model estimation
         lw = (x[1]*SL[1])/(x[0]*SL[0]+x[1]*SL[1])
         #within interval power model estimation
         pw = ((SP[1]**2)*x[1]**2)/(((SP[0]**2)*x[0]**2)+((SP[1]**2)*x[1]**2))
      case 1:
         smin = 0.125
         smax = 0.250
         lw = (x[1]*SL[2])/(x[0]*SL[1]+x[1]*SL[2])
         pw = ((SP[2]**2)*x[1]**2)/(((SP[1]**2)*x[0]**2)+((SP[2]**2)*x[1]**2))
      case 2:
         smin = 0.250
         smax = 0.375
         lw = (x[1]*SL[3])/(x[0]*SL[2]+x[1]*SL[3])
         pw = ((SP[3]**2)*x[1]**2)/(((SP[2]**2)*x[0]**2)+((SP[3]**2)*x[1]**2))
      case 3:
         smin = 0.375
         smax = 0.500
         lw = (x[1]*SL[4])/(x[0]*SL[3]+x[1]*SL[4])
         pw = ((SP[4]**2)*x[1]**2)/(((SP[3]**2)*x[0]**2)+((SP[4]**2)*x[1]**2))
      case 4:
         smin = 0.500
         smax = 0.625
         lw = (x[1]*SL[5])/(x[0]*SL[4]+x[1]*SL[5])
         pw = ((SP[5]**2)*x[1]**2)/(((SP[4]**2)*x[0]**2)+((SP[5]**2)*x[1]**2))
      case 5:
         smin = 0.625
         smax = 0.750
         lw = (x[1]*SL[6])/(x[0]*SL[5]+x[1]*SL[6])
         pw = ((SP[6]**2)*x[1]**2)/(((SP[5]**2)*x[0]**2)+((SP[6]**2)*x[1]**2))
      case 6:
         smin = 0.750
         smax = 0.875
         lw = (x[1]*SL[7])/(x[0]*SL[6]+x[1]*SL[7])
         pw = ((SP[7]**2)*x[1]**2)/(((SP[6]**2)*x[0]**2)+((SP[7]**2)*x[1]**2))
      case 7:
         smin = 0.875
         smax = 1.000
         lw = (x[1]*SL[0])/(x[0]*SL[7]+x[1]*SL[0])
         pw = ((SP[0]**2)*x[1]**2)/(((SP[7]**2)*x[0]**2)+((SP[0]**2)*x[1]**2))
         
   #compute the scaled value of the linear and power model estimation
   l = (lw * (smax-smin)) + smin
   p = (pw * (smax-smin)) + smin
   #combine into a single varible for easier use
   xp = [l, p]
   return xp


def linear_estimate(S0, S1, S2, S3, S4, S5, S6, S7, data):
   #within interval linear model estimation
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match data[2]:
      case 0:
         smin = 0
         smax = 0.125
         lw = (S1*data[1])/(S0*data[0]+S1*data[1])
      case 1:
         smin = 0.125
         smax = 0.250
         lw = (S3*data[1])/(S1*data[0]+S2*data[1])
      case 2:
         smin = 0.250
         smax = 0.375
         lw = (S3*data[1])/(S2*data[0]+S3*data[1])
      case 3:
         smin = 0.375
         smax = 0.500
         lw = (S4*data[1])/(S3*data[0]+S4*data[1])
      case 4:
         smin = 0.500
         smax = 0.625
         lw = (S5*data[1])/(S4*data[0]+S5*data[1])
      case 5:
         smin = 0.625
         smax = 0.750
         lw = (S6*data[1])/(S5*data[0]+S6*data[1])
      case 6:
         smin = 0.750
         smax = 0.875
         lw = (S7*data[1])/(S6*data[0]+S7*data[1])
      case 7:
         smin = 0.875
         smax = 1.000
         lw = (S0*data[1])/(S7*data[0]+S0*data[1])
   #compute the scaled value of the linear and power model estimation
   l = (lw * (smax-smin)) + smin
   return l

def power_estimate(S0, S1, S2, S3, S4, S5, S6, S7, data):
   #within interval linear model estimation
   #place variables for denoting the scaling level based on tactor pair loaction
   smax = 1
   smin = 0
   #scaling the data according to tactor pair location
   match data[2]:
      case 0:
         smin = 0
         smax = 0.125
         pw = ((S1**2)*(data[1]**2))/((S0**2)*(data[0]**2)+(S1**2)*(data[1]**2))
      case 1:
         smin = 0.125
         smax = 0.250
         pw = ((S2**2)*(data[1]**2))/((S1**2)*(data[0]**2)+(S2**2)*(data[1]**2))
      case 2:
         smin = 0.250
         smax = 0.375
         pw = ((S3**2)*(data[1]**2))/((S2**2)*(data[0]**2)+(S3**2)*(data[1]**2))
      case 3:
         smin = 0.375
         smax = 0.500
         pw = ((S4**2)*(data[1]**2))/((S3**2)*(data[0]**2)+(S4**2)*(data[1]**2))
      case 4:
         smin = 0.500
         smax = 0.625
         pw = ((S5**2)*(data[1]**2))/((S4**2)*(data[0]**2)+(S5**2)*(data[1]**2))
      case 5:
         smin = 0.625
         smax = 0.750
         pw = ((S6**2)*(data[1]**2))/((S5**2)*(data[0]**2)+(S6**2)*(data[1]**2))
      case 6:
         smin = 0.750
         smax = 0.875
         pw = ((S7**2)*(data[1]**2))/((S6**2)*(data[0]**2)+(S7**2)*(data[1]**2))
      case 7:
         smin = 0.875
         smax = 1.000
         pw = ((S0**2)*(data[1]**2))/((S7**2)*(data[0]**2)+(S0**2)*(data[1]**2))
   #compute the scaled value of the linear and power model estimation
   p = (pw * (smax-smin)) + smin
   return p

#optimization linear function
def linear_optimum(x, angle, response):
   #get parameters
   S0 = x[0]
   S1 = x[1]
   S2 = x[2]
   S3 = x[3]
   S4 = x[4]
   S5 = x[5]
   S6 = x[6]
   S7 = x[7]
   predicted_response = []
   error = []
   error_sq = 0
   j = 0
   for i in angle:
      Motor = Arduino_Magl(i)
      predicted_response.append(linear_estimate(S0, S1, S2, S3, S4, S5, S6, S7, Motor))
   
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
   S2 = x[2]
   S3 = x[3]
   S4 = x[4]
   S5 = x[5]
   S6 = x[6]
   S7 = x[7]
   error_sq = 0
   j = 0
   error = []
   predicted_response = []
   for i in angle:
      Motor = Arduino_Magl(i)
      predicted_response.append(power_estimate(S0, S1, S2, S3, S4, S5, S6, S7, Motor))

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
angles = []
response = []
normal_angles = []
normal_response = []
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
   angles.append(Data[i][1])
   response.append(Data[i][5])
   normal_angles.append(value_normalizing(Data[i][1]))
   normal_response.append(value_normalizing(Data[i][5]))
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


#get normal static and dynamic epsilon absolute decoding error.
conversions = 0
conversiond = 0
converter = 0
normal_static_epsilon = []
normal_dynamic_epsilon = []
normal_epsilon = []


while conversions <= len(normal_response_stat)-1:
   normal_static_error = normal_response_stat[conversions] - normal_angles_stat[conversions]
   if (normal_static_error > 0.5 or normal_static_error < -0.5):
      if (angles_stat[conversions] <= 180):
         abs_normal_static_error = np.absolute(normal_response_stat[conversions] - normal_angles_stat[conversions])
         normal_static_epsilon.append(abs_normal_static_error)
      elif (angles_stat[conversions] > 180):
         abs_normal_static_error = np.absolute(normal_response_stat[conversions] - normal_angles_stat[conversions])
         normal_static_epsilon.append(abs_normal_static_error)
   else:
      normal_static_epsilon.append(np.absolute(normal_static_error))
   conversions = conversions + 1

while conversiond <= len(normal_response_dyn)-1:
   normal_dyn_error = normal_response_dyn[conversiond] - normal_angles_dyn[conversiond]
   if (normal_dyn_error > 0.47 or normal_dyn_error < -0.47):
      if (angles_dyn[conversiond] <=180):
         abs_normal_dyn_error = np.absolute(normal_response_dyn[conversiond] - normal_angles_dyn[conversiond])
         normal_dynamic_epsilon.append(abs_normal_dyn_error)
      elif (angles_dyn[conversiond] > 180):
         abs_normal_dyn_error = np.absolute(normal_response_dyn[conversiond] - normal_angles_dyn[conversiond])
         normal_dynamic_epsilon.append(abs_normal_dyn_error)
   else:
      normal_dynamic_epsilon.append(np.absolute(normal_dyn_error))
   conversiond = conversiond + 1

while converter <= len(normal_response)-1:
   normal_error = normal_response[converter] - normal_angles[converter]
   normal_epsilon.append(np.absolute(normal_error))
   converter = converter + 1

#get normalized means and standard deviations of normal data
normal_static_mean = np.mean(normal_static_epsilon)
normal_dynamic_mean = np.mean(normal_dynamic_epsilon)
normal_static_std = np.std(normal_static_epsilon)
normal_dynamic_std = np.std(normal_dynamic_epsilon)


stat = [angles_stat, angles_statr]
#perform minimizing
guess = [5, 52, 14, 43, 10, 25, 35, 60]
#L-BFGS-B
#BFGS
#Nelder-Mead
#TNC
#SLSQP

bds = ((0.5, 100), (0.5, 100), (0.5, 100), (0.5, 100), (0.5, 100), (0.5, 100), (0.5, 100), (0.5, 100))
resultsl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds)

resultsp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds)

resultdl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds)

resultdp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds)

resultl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (normal_angles, normal_response), bounds = bds)

resultp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (normal_angles, normal_response), bounds = bds)
SL = resultl.x
SP = resultp.x

Ssl = resultsl.x
Ssp = resultsp.x
Sdl = resultdl.x
Sdp = resultdp.x

#compute decoding error for against the models using while loop
decode = 0
epsilonsl = []
epsilonsp = []
epsilondl = []
epsilondp = []
while decode <= len(normal_response_stat)-1:
   #computing static error
   valsd = Arduino_Magl(angles_stat[decode])
   estsd = intensity_estimate_adjusted(valsd, Ssl, Ssp)
   xllsd = estsd[0]
   xppsd = estsd[1]
   errsl = normal_response_stat[decode] - xllsd
   errsp = normal_response_stat[decode] - xppsd

   #wrapping of error on 0-1 scale for static linear
   if (errsl > 0.5 or errsl < -0.5):
      if (normal_angles_stat[decode] < 0.5):
         errorsl = normal_response_stat[decode] - xllsd
         epsilonsl.append(np.absolute(errorsl))
      elif(normal_angles_stat[decode] > 0.5):
         errorsl = normal_response_stat[decode] - xllsd
         epsilonsl.append(np.absolute(errorsl))
   else:
      epsilonsl.append(np.absolute(errsl))
   #wrapping error on 0-1 scale for static power
   if (errsp > 0.5 or errsp < -0.5):
      if (normal_angles_stat[decode] < 0.5):
         errorsp = normal_response_stat[decode] - xppsd
         epsilonsp.append(np.absolute(errorsp))
      elif(normal_angles_stat[decode] > 0.5):
         errorsp = normal_response_stat[decode] - xppsd
         epsilonsp.append(np.absolute(errorsp))
   else:
      epsilonsp.append(np.absolute(errsp))
   decode = decode+1

decode2 = 0
while decode2 <= len(normal_response_dyn)-1:
   #computing dynamic error
   valdd = Arduino_Magl(angles_dyn[decode2])
   estdd = intensity_estimate_adjusted(valdd, Sdl, Sdp)
   xlldd = estdd[0]
   xppdd = estdd[1]
   errdl = normal_response_dyn[decode2] - xlldd
   errdp = normal_response_dyn[decode2] - xppdd
   #wrapping of error on 0-1 scale for dynamic linear
   if (errdl > 0.5 or errdl < -0.5):
      if (angles_dyn[decode2] < 180):
         errordl = normal_response_dyn[decode2] - xlldd
         epsilondl.append(np.absolute(errordl))
      elif(angles_dyn[decode2] > 180):
         errordl = normal_response_dyn[decode2] - xlldd
         epsilondl.append(np.absolute(errordl))
   else:
      epsilondl.append(np.absolute(errdl))

   #wrapping error on 0-1 scale for dynamic power
   if (errdp > 0.5 or errdp < -0.5):
      if (angles_dyn[decode2] < 180):
         errordp = normal_response_dyn[decode2] - xppdd
         epsilondp.append(np.absolute(errordp))
      elif(angles_dyn[decode2] > 180):
         errordp = normal_response_dyn[decode2] - xppdd
         epsilondp.append(np.absolute(errordp))
   else:
      epsilondp.append(np.absolute(errdp))
   decode2 = decode2+1

decode3 = 0
decode4 = 0
epsilonsnal = []
epsilonsnap = []
epsilondnal = []
epsilondnap = []
while decode3 <= len(normal_response_stat)-1:
   #computing static error
   valsd = Arduino_Magl(angles_stat[decode3])
   estsd = intensity_estimate(valsd)
   xllsd = estsd[0]
   xppsd = estsd[1]
   errsl = normal_response_stat[decode3] - xllsd
   errsp = normal_response_stat[decode3] - xppsd

   #wrapping of error on 0-1 scale for static linear
   if (errsl > 0.5 or errsl < -0.5):
      if (angles_stat[decode3] < 180):
         errorsl = normal_response_stat[decode3] - xllsd
         epsilonsnal.append(np.absolute(errorsl))
      elif(angles_stat[decode3] > 180):
         errorsl = normal_response_stat[decode3] - xllsd
         epsilonsnal.append(np.absolute(errorsl))
   else:
      epsilonsnal.append(np.absolute(errsl))
   #wrapping error on 0-1 scale for static power
   if (errsp > 0.5 or errsp < -0.5):
      if (angles_stat[decode3] < 180):
         errorsp = normal_response_stat[decode3] - xppsd
         epsilonsnap.append(np.absolute(errorsp))
      elif(angles_stat[decode3] > 180):
         errorsp = normal_response_stat[decode3] - xppsd
         epsilonsnap.append(np.absolute(errorsp))
   else:
      epsilonsnap.append(np.absolute(errsp))
   decode3 = decode3+1


while decode4 <= len(normal_response_dyn)-1:
   #computing dynamic error
   valdd = Arduino_Magl(angles_dyn[decode4])
   estdd = intensity_estimate(valdd)
   xlldd = estdd[0]
   xppdd = estdd[1]
   errdl = normal_response_dyn[decode4] - xlldd
   errdp = normal_response_dyn[decode4] - xppdd
   #wrapping of error on 0-1 scale for dynamic linear
   if (errdl > 0.5 or errdl < -0.5):
      if (angles_dyn[decode4] < 180):
         errordl = normal_response_dyn[decode4] - xlldd
         epsilondnal.append(np.absolute(errordl))
      elif(angles_dyn[decode4] > 180):
         errordl = normal_response_dyn[decode4] - xlldd
         epsilondnal.append(np.absolute(errordl))
   else:
      epsilondnal.append(np.absolute(errdl))

   #wrapping error on 0-1 scale for dynamic power
   if (errdp > 0.5 or errdp < -0.5):
      if (angles_dyn[decode4] < 180):
         errordp = normal_response_dyn[decode4] - xppdd
         epsilondnap.append(np.absolute(errordp))
      elif(angles_dyn[decode4] > 180):
         errordp = normal_response_dyn[decode4] - xppdd
         epsilondnap.append(np.absolute(errordp))
   else:
      epsilondnap.append(np.absolute(errdp))
   decode4 = decode4+1

epsilonl = []
epsilonp = [] 
decode5 = 0
while decode5 <= len(normal_response)-1:
   #computing static error
   valsd = Arduino_Magl(angles[decode5])
   estsd = intensity_estimate_adjusted(valsd, SL, SP)
   xllsd = estsd[0]
   xppsd = estsd[1]
   errsl = normal_response[decode5] - xllsd
   errsp = normal_response[decode5] - xppsd

   #wrapping of error on 0-1 scale for static linear
   if (errsl > 0.5 or errsl < -0.5):
      if (angles[decode5] < 180):
         errorsl = normal_response[decode5] - xllsd
         epsilonl.append(np.absolute(errorsl))
      elif(angles[decode5] > 180):
         errorsl = normal_response[decode5] - xllsd
         epsilonl.append(np.absolute(errorsl))
   else:
      epsilonl.append(np.absolute(errsl))
   #wrapping error on 0-1 scale for static power
   if (errsp > 0.5 or errsp < -0.5):
      if (angles[decode5] < 180):
         errorsp = normal_response[decode5] - xppsd
         epsilonp.append(np.absolute(errorsp))
      elif(angles[decode5] > 180):
         errorp = normal_response[decode5] - xppsd
         epsilonp.append(np.absolute(errorsp))
   else:
      epsilonp.append(np.absolute(errsp))
   decode5 = decode5+1

#print(np.mean(epsilonl), np.mean(epsilonp))
#print("\n")
meansl = np.mean(epsilonsl)
meansp = np.mean(epsilonsp)
meandl = np.mean(epsilondl)
meandp = np.mean(epsilondp)

stdsl = np.std(epsilonsl)
stdsp = np.std(epsilonsp)
stddl = np.std(epsilondl)
stddp = np.std(epsilondp)

meansnal = np.mean(epsilonsnal)
meansnap = np.mean(epsilonsnap)
meandnal = np.mean(epsilondnal)
meandnap = np.mean(epsilondnap)

stdsnal = np.std(epsilonsnal)
stdsnap = np.std(epsilonsnap)
stddnal = np.std(epsilondnal)
stddnap = np.std(epsilondnap)

linear_epsilon_mean = np.mean(epsilonl)
power_epsilon_mean = np.mean(epsilonp)
linear_epsilon_std = np.std(epsilonl)
power_epsilon_std = np.std(epsilonp)

print("L Mean","L STD", "P Mean","P STD")
print("Adjusted Models")
print("\n")
print(meansl, stdsl ,meansp, stdsp)
print("\n")
print(meandl, stddl ,meandp, stddp)
print("\n")
print("Generic Models")
print(meansnal, stdsnal ,meansnap, stdsnap)
print("\n")
print(meandnal, stddnal ,meandnap, stddnap)
print("\n")

#print("Variances")
varsl = np.var(epsilonsl)
varsp = np.var(epsilonsp)
vardl = np.var(epsilondl)
vardp = np.var(epsilondp)
#print(varsl, varsp ,vardl, vardp)
#print("\n")
stlp = stats.ttest_ind(a=epsilonsl, b=epsilonsp, equal_var=True)
dtlp = stats.ttest_ind(a=epsilondl, b=epsilondp, equal_var=True)
sldl = stats.ttest_ind(a=epsilonsl, b=epsilondl, equal_var=True)
spdp = stats.ttest_ind(a=epsilonsp, b=epsilondp, equal_var=True)

print("Statistic Results")
print(stlp)
print("\n")
print(dtlp)
print("\n")
print("Adj Stat v Dyn Lin")
print(sldl)
print("\n")
print("Adj Stat v Dyn Power")
print(spdp)

snaldnal = stats.ttest_ind(a=epsilonsnal, b=epsilondnal, equal_var=True)
snapdnap = stats.ttest_ind(a=epsilonsnap, b=epsilondnap, equal_var=True)
snalsnap = stats.ttest_ind(a=epsilonsnal, b=epsilonsnap, equal_var=True)
dnaldnap = stats.ttest_ind(a=epsilondnal, b=epsilondnap, equal_var=True)
snalsl = stats.ttest_ind(a=epsilonsnal, b=epsilonsl, equal_var=True)
dnaldl = stats.ttest_ind(a=epsilondnal, b=epsilondl, equal_var=True)
snapsp = stats.ttest_ind(a=epsilonsnap, b=epsilonsp, equal_var=True)
dnapdp = stats.ttest_ind(a=epsilondnap, b=epsilondp, equal_var=True)
print("\n")
print("Gen Stat v Dyn Lin")
print(snaldnal)
print("\n")
print("Gen Stat v Dyn Power")
print(snapdnap)

print("\n")
print("Adj Stat v Gen Stat Lin")
print(snalsl)
print("\n")
print("Adj Dyn v Gen Dyn Lin")
print(dnaldl)
print("\n")
print("Adj Stat v Gen Stat Power")
print(snalsl)
print("\n")
print("Adj Dyn v Gen Dyn Power")
print(dnaldl)


#computing effect size
effgensdl = (meansnal - meandnal)/pool_std(epsilonsnal, epsilondnal)
effgensdp = (meansnap - meandnap)/pool_std(epsilonsnap, epsilondnap)
effadjsdl = (meansl - meandl)/pool_std(epsilonsl, epsilondl)
effadjsdp = (meansp - meandp)/pool_std(epsilonsp, epsilondp)
effadjslgensl = (meansl - meansnal)/pool_std(epsilonsl, epsilonsnal)
effadjspgensp = (meansp - meansnap)/pool_std(epsilonsp, epsilonsnap)
effadjdlgendl = (meandl - meandnal)/pool_std(epsilondl, epsilondnal)
effadjdpgendp = (meandp - meandnap)/pool_std(epsilondp, epsilondnap)

print("\n")
print("Effect Size Cohen's D")
print("\n")
print("Comparing Static and Dynamic within model")
print("Gen Stat v Dyn Lin")
print(effgensdl)
print("\n")
print("Gen Stat v Dyn Power")
print(effgensdp)
print("\n")
print("Adj Stat v Dyn Lin")
print(effadjsdl)
print("\n")
print("Adj Stat v Dyn Power")
print(effadjsdp)
print("\n")
print("Comparing Static and dynamic between models")
print("Adj Stat v Gen Stat Lin")
print(effadjslgensl)
print("\n")
print("Adj Stat v Gen Stat Power")
print(effadjspgensp)
print("\n")
print("Adj Dyn v Gen Dyn Lin")
print(effadjdlgendl)
print("\n")
print("Adj Dyn v Gen Dyn Power")
print(effadjdpgendp)

normal_static_V_gen_lin = stats.ttest_ind(a = normal_static_epsilon, b = epsilonsnal, equal_var =True)
normal_static_V_gen_power = stats.ttest_ind(a = normal_static_epsilon, b = epsilonsnap, equal_var = True)
normal_static_V_adj_lin = stats.ttest_ind(a = normal_static_epsilon, b = epsilonsl, equal_var = True)
normal_static_V_adj_power = stats.ttest_ind(a = normal_static_epsilon, b = epsilonsp, equal_var = True)

normal_dynamic_V_gen_lin = stats.ttest_ind(a = normal_dynamic_epsilon, b = epsilondnal, equal_var = True)
normal_dynamic_V_gen_power = stats.ttest_ind(a = normal_dynamic_epsilon, b = epsilondnap, equal_var = True)
normal_dynamic_V_adj_lin = stats.ttest_ind(a = normal_dynamic_epsilon, b = epsilondl, equal_var = True)
normal_dynamic_V_adj_power = stats.ttest_ind(a = normal_dynamic_epsilon, b = epsilondp, equal_var = True)

normal_static_V_normal_dynamic = stats.ttest_ind(a = normal_static_epsilon, b = epsilonsl, equal_var = True)
print("\n")
print("\n")
print("T-tests for comparing static models with actual static response")
print(normal_static_V_gen_lin)
print("\n")
print(normal_static_V_gen_power)
print("\n")
print(normal_static_V_adj_lin)
print("\n")
print(normal_static_V_adj_power)
print("\n")
print("T-tests for comparing dynamic models with actual dynamic response")
print(normal_dynamic_V_gen_lin)
print("\n")
print(normal_dynamic_V_gen_power)
print("\n")
print(normal_dynamic_V_adj_lin)
print("\n")
print(normal_dynamic_V_adj_power)
print("\n")
print("\n")
print(normal_static_V_normal_dynamic)
print("\n")
print("\n")
print("Means and STD of all data model")
print(normal_static_mean, normal_static_std)
print("\n")
print(normal_dynamic_mean, normal_dynamic_std)
print("\n")

fig = plt.figure("Figure 1")
plt.scatter(normal_angles_stat, normal_static_epsilon, linewidth = 2.0)
plt.scatter(normal_response_stat, epsilonsl, linewidth = 2.0, color = 'r')

plt.show()

