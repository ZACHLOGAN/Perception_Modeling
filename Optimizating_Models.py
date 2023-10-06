import csv
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize

#function for computing linear model estimation and error
def Arduino_MagL(x, angles):
        S0 = x[0]
        S1 = x[1]
        
        des_max = 360
        des_min = 0
        normalized = (response - des_min)/(des_max-des_min)

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

        I0 = Motor_Data[0]
        I1 = Motor_Data[1]
        Spot = Motor_Data[2]

        #place variables for denoting the scaling level based on tactor pair loaction
        smax = 1
        smin = 0
        #scaling the data according to tactor pair location
        match Spot:
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

        #within interval linear model estimation
        lw = (S1*I1)/(S0*I0+S1*I1)
        #within interval power model estimation
        l = (lw * (smax-smin)) + smin
        #combine into a single varible for easier use

        errsl = normalized - l
        
        #wrapping of error on 0-1 scale for static linear
        if (errsl > 0.5 or errsl < -0.5):
                if (response < 180):
                        error = 1 + normalized - l
                        epsilon.append(np.absolute(error))
                elif(response > 180):
                        error = 1 - response - l
                        epsilon = np.absolute(error)
        else:
                epsilon = np.absolute(errsl)
        
        return epsilon

#function for computing power model estimation and error
def Arduino_MagP(angle, response, S0, S1):
        des_max = 360
        des_min = 0
        normalized = (response - des_min)/(des_max-des_min)
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

        x = Motor_Data

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

        #within interval linear model estimation
        pw = (S1*(x[1]**2))/((S0*(x[0]**2))+(S1*(x[1]**2)))
        #within interval power model estimation
        p = (pw * (smax-smin)) + smin
        #combine into a single varible for easier use

        v = f(p, S0, S1)
        errsl = normalized - v

        #wrapping of error on 0-1 scale for static linear
        if (errsl > 0.5 or errsl < -0.5):
                if (response < 180):
                        error = 1 + normalized - v
                        epsilon.append(np.absolute(error))
        elif(response > 180):
                        error = 1 - response - v
                        epsilon.append(np.absolute(error))
        else:
                epsilon.append(np.absolute(errsl))
        
        return epsilon

#files needed to get participant response data
filetypes = {
    1:"alldata.csv"}
#load csv into list and delete headers and labels
Data = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
Data = np.delete(Data,0,0)
Data = np.delete(Data,0,1)
#seperate angles out into static and dynamic
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

  
res_statl = minimize(Arduino_MagL, angles_statr, 
