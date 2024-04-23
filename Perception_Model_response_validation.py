import csv
import random
import itertools
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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
#      1:"participant018_computed.csv"}

#set only one participant data type for testing methodology
Data = genfromtxt(DIR+filetypes[1],delimiter=',',dtype=float)
Data = np.delete(Data,0,0)
Data = np.delete(Data,0,1)

#angles used in experiment including 0 and 360 for completing model lines
given_angles = [15, 22.5, 50, 75, 105, 112.5, 140, 165, 195, 202.5, 220, 255, 285, 295.5, 310, 345]
angles = []

#seperate out only the condition, desired angle and participant response
i = 0
while i <= len(Data)-1:
		the_value = [Data[i][0], Data[i][1], Data[i][5]]
		angles.append(the_value)
		i = i + 1
#convert into a numpy array
needed_angles = np.array(angles)

#create the kfold object for spliting data into X sets of train and test data
kfold = KFold(n_splits = 10, random_state = 1, shuffle = True)

#variables for storing all of the final model coefficients
FSL = []
FSP = []
FSSL = []
FSSP = []
FSDL = []
FSDP = []

#variables for storing the means and std of the error calculations
Error_Means = []
Error_STD = []
#creates indicies of the train and test data for set number of splits
for train, test in kfold.split(needed_angles):
	#print(len(train), len(test))
	#variables for storing the different pieces needed for model optimization
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
	regular_angles = []
	#seperate the training data indexs into the correct data type
	i = 0
	while i <= len(train)-1:
			normal_angles.append(value_normalizing(needed_angles[train[i]][1]))
			regular_angles.append(needed_angles[train[i]][1])
			normal_response.append(value_normalizing(needed_angles[train[i]][2]))
			if (needed_angles[train[i]][0] == 0):
					normal_angles_stat.append(value_normalizing(needed_angles[train[i]][1]))
					normal_response_stat.append(value_normalizing(needed_angles[train[i]][2]))
					angles_stat.append(needed_angles[train[i]][1])
					angles_statr.append(needed_angles[train[i]][2])
			elif(needed_angles[train[i]][0] == 1):
					normal_angles_dyn.append(value_normalizing(needed_angles[train[i]][1]))
					normal_response_dyn.append(value_normalizing(needed_angles[train[i]][2]))
					angles_dyn.append(needed_angles[train[i]][1])
					angles_dynr.append(needed_angles[train[i]][2])
			i = i + 1
	#optimization of different perception models
	guess_0 = random.randint(2, 20)
	guess_1 = random.randint(2, 20)
	guess_2 = random.randint(2, 20)
	guess_3 = random.randint(2, 20)
	guess_4 = random.randint(2, 20)
	guess_5 = random.randint(2, 20)
	guess_6 = random.randint(2, 20)
	guess_7 = random.randint(2, 20)
	guess = [guess_0, guess_1, guess_2,  guess_3,  guess_4,  guess_5,  guess_6,  guess_7]
	bds = ((1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500), (1, 500))
	resultsl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds, options = {'ftol':1E-5})

	resultsp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_stat, normal_response_stat), bounds = bds, options = {'ftol':1E-5})

	resultdl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds, options = {'ftol':1E-5})

	resultdp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (angles_dyn, normal_response_dyn), bounds = bds, options = {'ftol':1E-5})

	resultl = optimize.minimize(linear_optimum, guess, method='L-BFGS-B', args = (regular_angles, normal_response), bounds = bds, options = {'ftol':1E-5})

	resultp = optimize.minimize(power_optimum, guess, method='L-BFGS-B', args = (regular_angles, normal_response), bounds = bds, options = {'ftol':1E-5})

	#pull coefficient values from optimization results
	SL = resultl.x
	SP = resultp.x

	Ssl = resultsl.x
	Ssp = resultsp.x
	Sdl = resultdl.x
	Sdp = resultdp.x

	#append solved coefficients to variables to look at later
	FSL.append(SL)
	FSP.append(SP)
	FSSL.append(Ssl)
	FSSP.append(Ssp)
	FSDL.append(Sdl)
	FSDP.append(Sdp)

	#compute the error of each of the models using the test data
	errsl = []
	errsp = []
	errdl = []
	errdp = []
	errl = []
	errp = []
	
	fullerrsl = []
	fullerrsp = []
	fullerrdl = []
	fullerrdp = []
	fullerrl = []
	fullerrp = []
	for given in test:
			desired = value_normalizing(needed_angles[given][1])
			data = Arduino_Magl(needed_angles[given][1])
			ps_response = intensity_estimate_adjusted(data, Ssl, Ssp)
			pd_response = intensity_estimate_adjusted(data, Sdl, Sdp)
			p_response = intensity_estimate_adjusted(data, SL, SP)
			sl_error = np.absolute(desired - ps_response[0])
			sp_error = np.absolute(desired - ps_response[1])
			dl_error = np.absolute(desired - pd_response[0])
			dp_error = np.absolute(desired - pd_response[1])
			l_error = np.absolute(desired - p_response[0])
			p_error = np.absolute(desired - p_response[1])
			errsl.append( sl_error )
			errsp.append( sp_error )
			errdl.append( dl_error )
			errdp.append( dp_error )
			errl.append( l_error )
			errp.append( p_error )
			fullerrsl.append([needed_angles[given][0], needed_angles[given][1], sl_error*360])
			fullerrsp.append([needed_angles[given][0], needed_angles[given][1], sp_error*360])
			fullerrdl.append([needed_angles[given][0], needed_angles[given][1], dl_error*360])
			fullerrdp.append([needed_angles[given][0], needed_angles[given][1], dp_error*360])
			fullerrl.append([needed_angles[given][0], needed_angles[given][1], l_error*360])
			fullerrp.append([needed_angles[given][0], needed_angles[given][1], p_error*360])

	meansl = np.mean(errsl)*360
	meansp = np.mean(errsp)*360
	meandl = np.mean(errdl)*360
	meandp = np.mean(errdp)*360
	meanl = np.mean(errl)*360
	meanp = np.mean(errp)*360
	
	varsl = np.std(errsl)*360
	varsp = np.std(errsp)*360
	vardl = np.std(errdl)*360
	vardp = np.std(errdp)*360
	varl = np.std(errl)*360
	varp = np.std(errp)*360

	mean_value = [meansl, meansp, meandl, meandp, meanl, meanp]
	std_value = [varsl, varsp, vardl, vardp, varl, varp]

	Error_Means.append(mean_value)
	Error_STD.append(std_value)
#print("SL","SP","DL","DP","L", "P")
#print("\n")
#print(Error_Means)
#print("\n")
#print(Error_STD)

ESSLM = []
ESSPM = []
ESDLM = []
ESDPM = []
ESLM = []
ESPM = []

ESSLS = []
ESSPS = []
ESDLS = []
ESDPS = []
ESLS = []
ESPS = []


e = 0
while e <= len(Error_Means)-1:
	ESSLM.append(Error_Means[e][0])
	ESSPM.append(Error_Means[e][1])
	ESDLM.append(Error_Means[e][2])
	ESDPM.append(Error_Means[e][3])
	ESLM.append(Error_Means[e][4])
	ESPM.append(Error_Means[e][5])

	ESSLS.append(Error_STD[e][0])
	ESSPS.append(Error_STD[e][1])
	ESDLS.append(Error_STD[e][2])
	ESDPS.append(Error_STD[e][3])
	ESLS.append(Error_STD[e][4])
	ESPS.append(Error_STD[e][5])
	
	e = e + 1

RSSLM = np.sum(ESSLM)/10
RSSPM = np.sum(ESSPM)/10
RSDLM = np.sum(ESDLM)/10
RSDPM = np.sum(ESDPM)/10
RSLM = np.sum(ESLM)/10
RSPM = np.sum(ESPM)/10

RSSLS = np.sum(ESSLS)/10
RSSPS = np.sum(ESSPS)/10
RSDLS = np.sum(ESDLS)/10
RSDPS = np.sum(ESDPS)/10
RSLS = np.sum(ESLS)/10
RSPS = np.sum(ESPS)/10

mean_all = [RSSLM, RSSPM, RSDLM, RSDPM, RSLM, RSPM]
std_all = [RSSLS, RSSPS, RSDLS, RSDPS, RSLS, RSPS]


print("\n")
print(RSSLM, RSSPM, RSDLM, RSDPM, RSLM, RSPM)
print("\n")
print(RSSLS, RSSPS, RSDLS, RSDPS, RSLS, RSPS)

#computing the means of model data accross angle and condition with pandas
dfsl = pd.DataFrame(fullerrsl, columns = ["Condition", "Target Angle", "Error"])
dfsp = pd.DataFrame(fullerrsp, columns = ["Condition", "Target Angle", "Error"])
dfdl = pd.DataFrame(fullerrdl, columns = ["Condition", "Target Angle", "Error"])
dfdp = pd.DataFrame(fullerrdp, columns = ["Condition", "Target Angle", "Error"])
dfl = pd.DataFrame(fullerrl, columns = ["Condition", "Target Angle", "Error"])
dfp = pd.DataFrame(fullerrp, columns = ["Condition", "Target Angle", "Error"])

msl = dfsl.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})
msp = dfsp.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})
mdl = dfdl.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})
mdp = dfdp.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})
ml = dfl.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})
mp = dfp.groupby(['Condition', 'Target Angle']).agg({'Error':['mean','std']})


al = [0, 1, 2, 3, 4, 5]
k = 2
combo = list(itertools.combinations(al,k))
print(combo)
for com in combo:
		test = stats.ttest_ind_from_stats(mean_all[com[0]], std_all[com[0]], 10, mean_all[com[1]], std_all[com[1]], 10, equal_var = True)
		print(test)
		print("\n")

#write the data of model error to csv file for use finding out the means via angle
"""
name = DIR+"total_errors.csv"
labels = ["Condition", "Target Angle", "Error"]
with open(name, 'w') as csvfile:
   testwriter = csv.writer(csvfile, delimiter = ',')
   testwriter.writerow(labels)

with open(name, 'a') as csvfile:
   testwriter = csv.writer(csvfile, delimiter = ',')
   testwriter.writerows()
"""
