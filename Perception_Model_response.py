import csv
import random
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
from statsmodels.stats.weightstats import ttost_ind
from sklearn.metrics import mean_squared_error

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
	
	smax = 1
	smin = 0
	#compute whole space value v
	l = (lw + x[2])/8
	p = (pw + x[2])/8
	#combine into a single varible for easier use
	xp = [l, p]
	return xp


def intensity_estimate_adjusted(x, SL, SP):
	#within interval linear model estimation
	#lw = (x[1]*SL[1])/(x[0]*SL[0]+x[1]*SL[1])
	#within interval power model estimation
	#pw = ((SP[1]**2)*x[1]**2)/(((SP[0]**2)*x[0]**2)+((SP[1]**2)*x[1]**2))
        #compute whole space value v
	match x[2]:
		#tactors 0-1
		case 0:
			lw = (x[1]*SL[1])/(x[0]*SL[0]+x[1]*SL[1])
			pw = ((SP[1]**2)*x[1]**2)/(((SP[0]**2)*x[0]**2)+((SP[1]**2)*x[1]**2))
		#tactors 1-2
		case 1:
			lw = (x[1]*SL[2])/(x[0]*SL[1]+x[1]*SL[2])
			pw = ((SP[2]**2)*x[1]**2)/(((SP[1]**2)*x[0]**2)+((SP[2]**2)*x[1]**2))
		#tactors 2-3
		case 2:
			lw = (x[1]*SL[3])/(x[0]*SL[2]+x[1]*SL[3])
			pw = ((SP[3]**2)*x[1]**2)/(((SP[2]**2)*x[0]**2)+((SP[3]**2)*x[1]**2))
		#tactors 3-4
		case 3:
			lw = (x[1]*SL[4])/(x[0]*SL[3]+x[1]*SL[4])
			pw = ((SP[4]**2)*x[1]**2)/(((SP[3]**2)*x[0]**2)+((SP[4]**2)*x[1]**2))
		#tactors 4-5
		case 4:
			lw = (x[1]*SL[5])/(x[0]*SL[4]+x[1]*SL[5])
			pw = ((SP[5]**2)*x[1]**2)/(((SP[4]**2)*x[0]**2)+((SP[5]**2)*x[1]**2))
		#tactors 5-6
		case 5:
			lw = (x[1]*SL[6])/(x[0]*SL[5]+x[1]*SL[6])
			pw = ((SP[6]**2)*x[1]**2)/(((SP[5]**2)*x[0]**2)+((SP[6]**2)*x[1]**2))
		#tactors 6-7
		case 6:
			lw = (x[1]*SL[7])/(x[0]*SL[6]+x[1]*SL[7])
			pw = ((SP[7]**2)*x[1]**2)/(((SP[6]**2)*x[0]**2)+((SP[7]**2)*x[1]**2))
		#tactors 7-0
		case 7:
			lw = (x[1]*SL[0])/(x[0]*SL[7]+x[1]*SL[0])
			pw = ((SP[0]**2)*x[1]**2)/(((SP[7]**2)*x[0]**2)+((SP[0]**2)*x[1]**2))

	l = (lw + x[2])/8
	p = (pw + x[2])/8
	#combine into a single varible for easier use
	xp = [l, p]
	return xp


def linear_estimate(S0, S1, S2, S3, S4, S5, S6, S7, x):
	N = 8
	M = 8
	SL = [S0, S1, S2, S3, S4, S5, S6, S7]
	#within interval linear model estimation
	match x[2]:
		#tactors 0-1
		case 0:
			lw = (x[1]*SL[1])/(x[0]*SL[0]+x[1]*SL[1])
		#tactors 1-2
		case 1:
			lw = (x[1]*SL[2])/(x[0]*SL[1]+x[1]*SL[2])
		#tactors 2-3
		case 2:
			lw = (x[1]*SL[3])/(x[0]*SL[2]+x[1]*SL[3])
		#tactors 3-4
		case 3:
			lw = (x[1]*SL[4])/(x[0]*SL[3]+x[1]*SL[4])
		#tactors 4-5
		case 4:
			lw = (x[1]*SL[5])/(x[0]*SL[4]+x[1]*SL[5])
		#tactors 5-6
		case 5:
			lw = (x[1]*SL[6])/(x[0]*SL[5]+x[1]*SL[6])
		#tactors 6-7
		case 6:
			lw = (x[1]*SL[7])/(x[0]*SL[6]+x[1]*SL[7])
		#tactors 7-0
		case 7:
			lw = (x[1]*SL[0])/(x[0]*SL[7]+x[1]*SL[0])
	
	#find value, v for the whole space
	l = (lw + x[2])/M
	
	return l

def power_estimate(S0, S1, S2, S3, S4, S5, S6, S7, x):
		N = 8
		M = 8
		SP = [S0, S1, S2, S3, S4, S5, S6, S7]
		match x[2]:
			#tactors 0-1
			case 0:
				pw = ((SP[1]**2)*x[1]**2)/(((SP[0]**2)*x[0]**2)+((SP[1]**2)*x[1]**2))
			#tactors 1-2
			case 1:
				pw = ((SP[2]**2)*x[1]**2)/(((SP[1]**2)*x[0]**2)+((SP[2]**2)*x[1]**2))
			#tactors 2-3
			case 2:
				pw = ((SP[3]**2)*x[1]**2)/(((SP[2]**2)*x[0]**2)+((SP[3]**2)*x[1]**2))
			#tactors 3-4
			case 3:
				pw = ((SP[4]**2)*x[1]**2)/(((SP[3]**2)*x[0]**2)+((SP[4]**2)*x[1]**2))
			#tactors 4-5
			case 4:
				pw = ((SP[5]**2)*x[1]**2)/(((SP[4]**2)*x[0]**2)+((SP[5]**2)*x[1]**2))
			#tactors 5-6
			case 5:
				pw = ((SP[6]**2)*x[1]**2)/(((SP[5]**2)*x[0]**2)+((SP[6]**2)*x[1]**2))
			#tactors 6-7
			case 6:
				pw = ((SP[7]**2)*x[1]**2)/(((SP[6]**2)*x[0]**2)+((SP[7]**2)*x[1]**2))
			#tactors 7-0
			case 7:
				pw = ((SP[0]**2)*x[1]**2)/(((SP[7]**2)*x[0]**2)+((SP[0]**2)*x[1]**2))
	
		p = (pw + x[2])/M
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
		
		error.append(err**2)
		j = j+1
			
	error_sq = sum(error)/len(response)
	
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
		error.append(err**2)
		j = j+1
			
	error_sq = sum(error)/len(response)
	
	return error_sq


#file directory for data files
DIR = "/Users/Zach/Documents/Modeling_for_Perception/"
#all filetypes

filetypes = {
	 1:"clean_alldata.csv"}

#filetypes = {
#        1:"participant018_computed.csv"}

#set only one participant data type for testing methodology
Data = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
Data = np.delete(Data,0,0)
Data = np.delete(Data,0,1)

#angles used in experiment including 0 and 360 for completing model lines
given_angles = [15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345]

#normalizing angles from static and dynamic trials split into desired and response process using while loops
angles = []
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



#perform minimizing
#L-BFGS-B
#BFGS
#Nelder-Mead
#TNC
#SLSQP
guess_0 = random.randint(1, 150)
guess_1 = random.randint(1, 150)
guess_2 = random.randint(1, 150)
guess_3 = random.randint(1, 150)
guess_4 = random.randint(1, 150)
guess_5 = random.randint(1, 150)
guess_6 = random.randint(1, 150)
guess_7 = random.randint(1, 150)
guess = [guess_0, guess_1, guess_2,  guess_3,  guess_4,  guess_5,  guess_6,  guess_7]
bds = ((1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500))
resultsl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds, options = {'ftol':1E-20})

resultsp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds, options = {'ftol':1E-20})

resultdl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds, options = {'ftol':1E-20})

resultdp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds, options = {'ftol':1E-20})

resultl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (normal_angles, normal_response), bounds = bds, options = {'ftol':1E-20})

resultp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (normal_angles, normal_response), bounds = bds, options = {'ftol':1E-20})

#pull coefficient values from optimization results
SL = resultl.x
SP = resultp.x

Ssl = resultsl.x
Ssp = resultsp.x
Sdl = resultdl.x
Sdp = resultdp.x


#check model success
errsl = []
errsp = []
errdl = []
errdp = []
errl = []
errp = []

for given in given_angles:
        desired = value_normalizing(given)
        data = Arduino_Magl(given)
        ps_response = intensity_estimate_adjusted(data, Ssl, Ssp)
        pd_response = intensity_estimate_adjusted(data, Sdl, Sdp)
        p_response = intensity_estimate_adjusted(data, SL, SP)
        errsl.append( np.absolute(desired - ps_response[0]) )
        errsp.append( np.absolute(desired - ps_response[1]) )
        errdl.append( np.absolute(desired - pd_response[0]) )
        errdp.append( np.absolute(desired - pd_response[1]) )
        errl.append( np.absolute(desired - p_response[0]) )
        errp.append( np.absolute(desired - p_response[1]) )

        

meansl = np.mean(errsl)
meansp = np.mean(errsp)
meandl = np.mean(errdl)
meandp = np.mean(errdp)
meanl = np.mean(errl)
meanp = np.mean(errp)

varsl = np.std(errsl)
varsp = np.std(errsp)
vardl = np.std(errdl)
vardp = np.std(errdp)
varl = np.std(errl)
varp = np.std(errp)

print(meansl*360, meansp*360, meandl*360, meandp*360, meanl*360, meanp*360)
print("\n")
print(varsl*360, varsp*360, vardl*360, vardp*360, varl*360, varp*360)
print("\n")


errgsl = []
errgsp = []
errgdl = []
errgdp = []

m = 0
while m <= len(angles_statr)-1:
        desired = value_normalizing(angles_statr[m])
        data = Arduino_Magl(angles_stat[m])
        s_response = intensity_estimate(data)
        errgsl.append( np.absolute( desired - s_response[0] ) )
        errgsp.append( np.absolute( desired - s_response[1] ) )

        m = m + 1

m = 0      
while m <= len(angles_dynr)-1:
        desired = value_normalizing(angles_dynr[m])
        data = Arduino_Magl(angles_dyn[m])
        d_response = intensity_estimate(data)
        errgdl.append( np.absolute( desired - d_response[0] ) )
        errgdp.append( np.absolute( desired - d_response[1] ) )
        m = m + 1

meangsl = np.mean(errgsl)
meangsp = np.mean(errgsp)
meangdl = np.mean(errgdl)
meangdp = np.mean(errgdp)

print(meangsl*360, meangsp*360, meangdl*360, meangdp*360)

"""
slsp = stats.ttest_ind(a=errsl, b=errsp, equal_var=True)
sldl = stats.ttest_ind(a=errsl, b=errdl, equal_var=True)
sldp = stats.ttest_ind(a=errsl, b=errdp, equal_var=True)
spdl = stats.ttest_ind(a=errsp, b=errdl, equal_var=True)
spdp = stats.ttest_ind(a=errsp, b=errdp, equal_var=True)
dldp = stats.ttest_ind(a=errdl, b=errdp, equal_var=True)
lp = stats.ttest_ind(a=errl, b=errp, equal_var=True)


mean_all = [0.003348377582824259, 0.013443236942411894, 0.00038244451401060406, 0.011968183948508834, 0.00493667961127498, 0.011947919600535841]
std_all = [0.00098591486168942, 0.004801176763106432, 0.00011265734131169221, 0.006011797047211558, 0.0014538968267478103, 0.006042998439521279]

mean_18 = [0.004586725520506802, 0.011998564345836705, 7.691086140733296e-05, 0.011900110300902262, 0.004203324824885427, 0.011884504757496533]
std_18 = [0.0013507312824869713, 0.005840295238064471, 2.2658127962934207e-05, 0.006057105706264289, 0.001237745359389924, 0.006093726466721735]

i = 0
while i <= len(mean_all)-1:
        test = stats.ttest_ind_from_stats(mean_all[i], std_all[i], len(errsl), mean_18[i], std_18[i], len(errsl), equal_var = True)
        print(test)
        print("\n")

        i = i + 1
"""
