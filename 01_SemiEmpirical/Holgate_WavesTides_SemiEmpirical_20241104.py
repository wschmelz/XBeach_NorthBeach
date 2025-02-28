import sys
import os
import time
import numpy
import glob
import datetime
import MCMC_20240917
from numba import njit
import scipy
from scipy import stats
import loess2D
import matplotlib.pyplot as plt

print(sys.version)

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
print(str(__file__))
print(time.strftime("%H:%M:%S"))

buoy_num = 44065
station_num = 8531680
tide_dir = wkspc + f"01_Station{station_num}/"
wave_dir = wkspc + f"00_Buoy{buoy_num}/"
vol_wkspc = wkspc + '02_TopoData/'

ero_file = wkspc + 'ero_output.csv'
ero_output = numpy.genfromtxt(ero_file, delimiter=',')
w1 = numpy.where(numpy.isnan(ero_output[:,1]))[0]
w2 = numpy.where(numpy.isnan(ero_output[:,1])==False)[0]
print(len(w1))
print(len(w2))
stde_use = numpy.std(ero_output[w2,1])/100.
print(stde_use)
ero_output[w1,1] = numpy.mean(ero_output[w2,1]) + numpy.random.normal(loc=0.,scale=stde_use,size=len(w1))

tide_file = wkspc + 'tide_output.csv'
tide_output = numpy.genfromtxt(tide_file, delimiter=',')
wave_file = wkspc + 'wave_output.csv'
wave_output = numpy.genfromtxt(wave_file, delimiter=',')

date_list_file = wkspc + 'date_list.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]
date_list = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])
date_list1 = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

date_list_file = vol_wkspc + 'date_list_topo.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]
date_list_topo = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

year_tmp = 2020
month_tmp = 10
day_tmp = 17
hour_tmp = 0
start_time = datetime.datetime(2009, 9, 15, 0) 
time_datum_start = datetime.datetime(2009, 9, 15, 0) 
year_tmp = 2021
month_tmp = 6
day_tmp = 1
hour_tmp = 0
time_datum_last_data = date_list_topo[-1]
year_tmp = 2020
month_tmp = 1
day_tmp = 1
hour_tmp = 0
time_datum_first_data = date_list_topo[0]
print(time_datum_start)
print(time_datum_last_data)
print(time_datum_first_data)
time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)
w_time = numpy.where((ero_output[:, 0] >= 0) & (ero_output[:, 0] <= time_tmp_last))[0]
dt = numpy.mean(ero_output[1:,0] - ero_output[0:-1,0])

@njit
def semi_empirical_1(theta, data_series1, dt):
    reim_sum = numpy.cumsum(theta[0] * (data_series1[w_time] - theta[1])) * dt
    return reim_sum

@njit
def semi_empirical_2(theta, data_series1, mean_val, dt):
    T_0 = numpy.zeros_like(data_series1)
    T_0[0] = mean_val
    for si_index in range(1, len(data_series1)):
        T_0[si_index] = T_0[si_index-1] + ((data_series1[si_index] - T_0[si_index-1]) / numpy.absolute(theta[1]))
    reim_sum = numpy.cumsum(theta[0] * (data_series1[w_time] - T_0[w_time])) * dt
    return reim_sum, T_0[w_time]

file_out = vol_wkspc + 'sh8_areas.csv'
volumes_tmp = numpy.reshape(numpy.genfromtxt(file_out, delimiter=','),(-1,2))
volumes = numpy.reshape(volumes_tmp[:,0].copy(),(-1,1))
print(volumes)

survey_volumes = numpy.zeros((20, 2))
date_list_vols = []
w_output = numpy.zeros(20, dtype="int")

for n in range(20):
    time_datum = date_list_topo[n]
    survey_volumes[n, 0] = (time_datum - time_datum_start).total_seconds() / (3600.0*24.0)
    survey_volumes[n, 1] = volumes[n,0]
    date_list_vols.append([time_datum,])
    print(numpy.where(ero_output[w_time, 0] == survey_volumes[n, 0])[0])
    w_output[n] = numpy.where(ero_output[w_time, 0] == survey_volumes[n, 0])[0]

mean_error = 50.0
print("Error:", mean_error)
print("Survey volumes:", survey_volumes)

theta_guess = numpy.array([
    2.61616172e-01,
    -5.57405367e-01,
    4.58619446e-03,
    2.13668758e+03,
    1.40370355e+00,
    1.06117782e+02,
    1.74245288e+01,
    1.61575578e-01,
    1.14932291e-07,
    -1.36383127e+01,
    -8.13691666e+00,
    6.62490883e+01,
    1.18873205e+01,
    3.40918177e+01
])
theta_priors = numpy.array([
    1.,
    -1.5,
    1.,
    24.*14.,
    1.,
    24.*14.,
    10.,
    0.1,
    1.e-07,
    -20.,
    -5.,
    120.,
    12.,
    5.0
])
stepsizes = (numpy.absolute(theta_priors))/100.
ero_output_tmp = ero_output.copy()

def function(theta):
    global ero_output_tmp
    theta1 = theta[0:2]
    theta2 = theta[2:4]
    theta3 = theta[4:6]
    theta4 = theta[6:9]
    theta5 = theta[9]
    ero_output_tmp = ero_output.copy()
    w1 = numpy.where(ero_output[:,1]<theta5)[0]
    ero_output_tmp[w1,1]=theta5*1.0
    semi_empirical1_out = semi_empirical_1(theta1, ero_output_tmp[:,1], dt)
    semi_empirical2_out = semi_empirical_2(theta2, ero_output_tmp[:,1], numpy.mean(ero_output_tmp[:,1]), dt)
    semi_empirical3_out = semi_empirical_2(theta3, ero_output_tmp[:,1], numpy.mean(ero_output_tmp[:,1]), dt)
    semi_empirical4 = theta4[0] + (theta4[1]*ero_output_tmp[w_time,0]) + (theta4[2]*(ero_output_tmp[w_time,0]**2.))
    semi_empirical = semi_empirical1_out + semi_empirical2_out[0] + semi_empirical3_out[0] + semi_empirical4
    return semi_empirical[w_output], semi_empirical, semi_empirical2_out[1], theta1[1]

res = scipy.stats.linregress(survey_volumes[:, 0], survey_volumes[:, 1])
print("result", res)
b1 = res.slope
b0 = res.intercept
b1_se = res.stderr
b0_se = res.intercept_stderr
r = res.rvalue
r2 = r**2.
print("slope:", round(b1,3))
print("intercept:", round(b0,3))
print("slope se:", round(b1_se,3))
training_data = survey_volumes[:, 1]
MCMC_iters, burn_in = 30000,5000

output_matrix_A, loglik_output = MCMC_20240917.MC_fit(function, training_data, theta_guess, theta_priors, stepsizes, MCMC_iters)
filename_base = wkspc + '03_Model_output/'
numpy.savetxt(filename_base + "posterior_params.txt", output_matrix_A, delimiter=',')
numpy.savetxt(filename_base + "loglik_output.txt", loglik_output, delimiter=',')

filename_base = wkspc + '03_Model_output/'
filename_A = filename_base + "posterior_params.txt"
output_matrix_A = numpy.genfromtxt(filename_A,delimiter=',')
filename_ll = filename_base + "loglik_output.txt"
loglik_output = numpy.genfromtxt(filename_ll,delimiter=',')

output_matrix_A = output_matrix_A[burn_in:,:]
theta_guess_mean = numpy.mean(output_matrix_A,axis=0)
print(theta_guess_mean)
MCMC_samples = 100
sample_indices = numpy.arange(0,len(output_matrix_A[:,0]),1)
numpy.random.shuffle(sample_indices)

function_out,function_out2,function_out3,function_out5 = function(output_matrix_A[0,:])
output_matrix_A2 = numpy.zeros((MCMC_samples,len(function_out2)))
output_matrix_A3 = numpy.zeros((MCMC_samples,len(function_out3)))
output_matrix_A5 = numpy.zeros((MCMC_samples,1))

for n in range(MCMC_samples):
    theta_sample = output_matrix_A[sample_indices[n],:]
    function_out,function_out2,function_out3,function_out5 = function(theta_sample)
    output_matrix_A2[n,:] = function_out2
    output_matrix_A3[n,:] = function_out3
    output_matrix_A5[n,:] = function_out5

burn_in = 0
fig = plt.figure(2, figsize=(16,16))
ax1 = plt.subplot(411)
T_0_mean = numpy.mean(output_matrix_A3,axis=0)
a2_mean = numpy.mean(output_matrix_A5,axis=0)
q_95 = numpy.quantile(tide_output[:,1],.975)
ax1.plot(date_list, tide_output[:,1], linewidth=1., color="dodgerblue", label="Sandy Hook WL")
label1 = "97.5%ile WL\n" + str(numpy.round(q_95,1)) + " ft rel. MLLW"
d1 = datetime.datetime(2009, 1, 1, 0) 
d2 = datetime.datetime(2019, 1, 1, 0) 
ax1.set_xlim(d1,d2)
ax1.set_xticklabels([])
ax1.grid()
ax1.set_ylabel("Sandy Hook WL\n(m. rel to MLLW)")

ax1 = plt.subplot(412)
q_95 = numpy.quantile(wave_output[:,1],.975)
ax1.plot(date_list, wave_output[:,1], linewidth=1., color="k")
label1 = "97.5%ile SWH\n" + str(numpy.round(q_95,1)) + " m"
ax1.set_xlim(d1,d2)
ax1.set_xticklabels([])
ax1.grid()
ax1.set_ylabel("NOAA buoy 440915\nsig. wave height (m.)")

ax1 = plt.subplot(413)
T_0_mean = numpy.mean(output_matrix_A3,axis=0)
a2_mean = numpy.mean(output_matrix_A5,axis=0)
q_95 = -7.5
ax1.plot(date_list, ero_output[:,1], linewidth=1., color="gray")
ax1.set_xlim(d1,d2)
ax1.set_xticklabels([])
ax1.grid()
ax1.legend(loc=3)
ax1.set_ylabel("Model predicted erosion\n(m² over 48h)")

date_list = date_list[w_time]
ax2 = plt.subplot(414)
semiempirical_mean = numpy.mean(output_matrix_A2,axis=0)
semiempirical_high2 = numpy.quantile(output_matrix_A2,0.975,axis=0)
semiempirical_low2 = numpy.quantile(output_matrix_A2,0.025,axis=0)
semiempirical_high = numpy.quantile(output_matrix_A2,0.16,axis=0)
semiempirical_low = numpy.quantile(output_matrix_A2,0.84,axis=0)
label2 = "$\int_{t_0}^{t} a_{1}(WL(t) - a_{2}) dt$ + " + "$\int_{t_0}^{t} a_{3}(WL(t) - WL_{0}(t)) dt$"
label_2_5 = ";\n $dWL_{0}/dt$"
a1_mean = numpy.mean(output_matrix_A[burn_in:,1],axis=0)
a1_std = numpy.std(output_matrix_A[burn_in:,1],axis=0)
a2_mean = numpy.mean(output_matrix_A[burn_in:,0],axis=0)
a2_std = numpy.std(output_matrix_A[burn_in:,0],axis=0)
a_mean = numpy.mean(output_matrix_A[burn_in:,3],axis=0)
a_std = numpy.std(output_matrix_A[burn_in:,3],axis=0)
k_mean = numpy.mean(output_matrix_A[burn_in:,4],axis=0)
k_std = numpy.std(output_matrix_A[burn_in:,4],axis=0)
err_mean = numpy.mean(output_matrix_A[burn_in:,-1],axis=0)
ax2.plot(date_list, semiempirical_mean, linewidth=1.5, color="gray", label="$f_{geo}(t)$")
ax2.plot(date_list, semiempirical_high2, linewidth=0.5, linestyle=":", color="gray", label="95% CI")
ax2.plot(date_list, semiempirical_high, linewidth=1.0, linestyle="--", color="gray")
ax2.plot(date_list, semiempirical_low, linewidth=1.0, linestyle="--", color="gray")
ax2.plot(date_list, semiempirical_low2, linewidth=0.5, linestyle=":", color="gray")

for n in range(len(w_output)):
    ax2.plot([date_list[w_output[n]], date_list[w_output[n]]],
             [training_data[n]-err_mean, training_data[n]+err_mean],
             linewidth=1.0, color="k")
    ax2.plot([date_list[w_output[n]], date_list[w_output[n]]],
             [training_data[n]-2.*err_mean, training_data[n]+2.*err_mean],
             linewidth=0.0, marker="_", color="k")

ax2.plot(date_list[w_output], training_data, linewidth=0.0, marker="o", color="k", label="Surveys")
events_real = numpy.array([1, 3, 2, 3, 3, 3, 2, 2, 2, 2])
events_prior = (events_real*0.0) + 0.5
num_years = 10
event_counts = numpy.zeros(num_years, dtype=numpy.int64)
date_list_storms = []
par_09_mean = numpy.mean(output_matrix_A[burn_in:,9],axis=0)
par_10_mean = numpy.mean(output_matrix_A[burn_in:,10],axis=0)
par_11_mean = numpy.mean(output_matrix_A[burn_in:,11],axis=0)
par_12_mean = numpy.mean(output_matrix_A[burn_in:,12],axis=0)
date_years = numpy.array([date.year for date in date_list], dtype=numpy.int64)

def compute_event_counts(function_out2, w_time, ero_output, date_years, event_counts):
    ero_output_tmp = ero_output.copy()
    w1_tmp = numpy.where(ero_output[:,1]<par_09_mean)[0]
    ero_output_tmp[w1_tmp,1]=par_09_mean*1.0
    low_val = function_out2[-1]
    for n2 in range(len(function_out2)):
        idx = len(function_out2) - (n2 + 1)
        valu = function_out2[idx]
        if valu < low_val:
            low_val = valu
            ero_idx_start = max(idx - int(par_11_mean), 0)
            ero_idx_end = min(idx + int(par_12_mean), len(w_time))
            semi_idx_start = max(idx - int(par_11_mean), 0)
            semi_idx_end = min(idx + int(par_11_mean), len(function_out2))
            ero_min = numpy.min(ero_output[w_time[ero_idx_start:ero_idx_end], 1])
            semi_min = numpy.min(function_out2[semi_idx_start:semi_idx_end])
            if (ero_min < par_10_mean) and (valu <= semi_min):
                event_year = date_years[idx]
                year_index = event_year - 2009
                if 0 <= year_index < num_years:
                    date_list_storms.append([date_list[idx]])
                    ax2.plot([date_list[idx], date_list[idx]], [valu, valu],
                             marker="o", linewidth=0.0, color="red")
                    event_counts[year_index] += 1
    return event_counts

event_counts_vector = compute_event_counts(semiempirical_mean, w_time, ero_output, date_years, event_counts*0)
ax2.plot([0,0],[-1000,-1000], marker="o", linewidth=0.0, color="red",
         label=f"Model predicted preserved\nerosion surfaces (n={str(int(numpy.sum(event_counts_vector)))})")
ax2.legend(loc=2,fontsize=6)
ax2.set_ylim(-50,350)
ax2.grid()
ax2.set_xlim(d1,d2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Volume change\n(m$^3$/m of shoreline)")
plt.tight_layout()
pltname = wkspc + "SemiEmpirical_Holgate_TMP.png"
plt.savefig(pltname, dpi=300)
plt.close()

MCMC_iters, burn_in = 100000,10000

output_matrix_A, loglik_output = MCMC_20240917.MC_fit(function, training_data, theta_guess_mean, theta_priors, stepsizes, MCMC_iters)
filename_base = wkspc + '03_Model_output/'
numpy.savetxt(filename_base + "posterior_params.txt", output_matrix_A, delimiter=',')
numpy.savetxt(filename_base + "loglik_output.txt", loglik_output, delimiter=',')

filename_base = wkspc + '03_Model_output/'
filename_A = filename_base + "posterior_params.txt"
output_matrix_A = numpy.genfromtxt(filename_A,delimiter=',')
filename_ll = filename_base + "loglik_output.txt"
loglik_output = numpy.genfromtxt(filename_ll,delimiter=',')
output_matrix_A = output_matrix_A[burn_in:,:]
burn_in = 0
theta_guess_mean = numpy.mean(output_matrix_A,axis=0)
MCMC_samples = 10000
sample_indices = numpy.arange(0,len(output_matrix_A[:,0]),1)
numpy.random.shuffle(sample_indices)
function_out,function_out2,function_out3,function_out5 = function(output_matrix_A[0,:])
output_matrix_A2 = numpy.zeros((MCMC_samples,len(function_out2)))
output_matrix_A3 = numpy.zeros((MCMC_samples,len(function_out3)))
output_matrix_A5 = numpy.zeros((MCMC_samples,1))

for n in range(MCMC_samples):
    theta_sample = output_matrix_A[sample_indices[n],:]
    function_out,function_out2,function_out3,function_out5 = function(theta_sample)
    output_matrix_A2[n,:] = function_out2
    output_matrix_A3[n,:] = function_out3
    output_matrix_A5[n,:] = function_out5

par_09_mean = numpy.mean(output_matrix_A[burn_in:,9],axis=0)
par_10_mean = numpy.mean(output_matrix_A[burn_in:,10],axis=0)
par_11_mean = numpy.mean(output_matrix_A[burn_in:,11],axis=0)
par_12_mean = numpy.mean(output_matrix_A[burn_in:,12],axis=0)
err_mean = numpy.mean(output_matrix_A[burn_in:, -1], axis=0)

plt.rcParams.update({'font.size': 6})
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
fig = plt.figure(2, figsize=(6.5, 6.5))
ax1 = plt.subplot(411)
T_0_mean = numpy.mean(output_matrix_A3, axis=0)
a2_mean = numpy.mean(output_matrix_A5, axis=0)
q_95 = (6.92 - 2.82) / 3.28084
ax1.plot(date_list1, (tide_output[:, 1] - 2.82) / 3.28084, linewidth=0.75, color="dodgerblue")
label1 = "Storm threshold WL\n" + str(numpy.round(q_95, 2)) + " m"
ax1.plot([date_list[0], date_list[-1]], [q_95, q_95], linewidth=1.0, color="red", label=label1)
ax1.text(0.0125, 0.95, labels[0], transform=ax1.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')
ax1.legend(fontsize=6)
d1 = datetime.datetime(2009, 1, 1, 0) 
d2 = datetime.datetime(2019, 1, 1, 0)
ax1.set_xlim(d1, d2)
ax1.set_xticklabels([])
ax1.grid()
ax1.set_ylabel("Sandy Hook WL\n(m rel to NAVD88)", fontsize=6)

ax2 = plt.subplot(412)
q_95 = 3.78
ax2.plot(date_list1, wave_output[:, 1], linewidth=0.75, color="k")
label1 = "Storm threshold SWH:\n" + str(numpy.round(q_95, 2)) + " m"
ax2.plot([date_list[0], date_list[-1]], [q_95, q_95], linewidth=1.0, color="red", label=label1)
ax2.legend(fontsize=6)
ax2.set_xlim(d1, d2)
ax2.set_xticklabels([])
ax2.grid()
ax2.set_ylabel("NOAA buoy 440915\nsig. wave height (m)", fontsize=6)
ax2.text(0.0125, 0.95, labels[1], transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')

ax3 = plt.subplot(413)
q_95 = -8.0
label1 = "Storm erosion threshold:\n" + str(numpy.round(q_95, 1)) + " m² over 48h"
ax3.plot(date_list1, ero_output[:, 1], linewidth=0.75, color="gray")
ax3.plot([date_list[0], date_list[-1]], [q_95, q_95], linewidth=1.0, color="red", label=label1)
ax3.set_xlim(d1, d2)
ax3.set_xticklabels([])
ax3.grid()
ax3.legend(loc=4, fontsize=6)
ax3.set_ylabel("XBeach model predicted\nerosion (m² over 48h)", fontsize=6)
ax3.text(0.0125, 0.1, labels[2], transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')

semiempirical_mean = numpy.mean(output_matrix_A2, axis=0)
semiempirical_high2 = numpy.quantile(output_matrix_A2, 0.975, axis=0)
semiempirical_low2 = numpy.quantile(output_matrix_A2, 0.025, axis=0)
semiempirical_high = numpy.quantile(output_matrix_A2, 0.84, axis=0)
semiempirical_low = numpy.quantile(output_matrix_A2, 0.16, axis=0)
ax4 = plt.subplot(414)
ax4.plot(date_list, semiempirical_mean, linewidth=0.75, color="gray", label="$f_{geo}(t)$")
ax4.plot(date_list, semiempirical_high2, linewidth=0.1, linestyle=":", color="gray", label="95% CI")
ax4.plot(date_list, semiempirical_high, linewidth=0.25, linestyle="--", color="gray")
ax4.plot(date_list, semiempirical_low, linewidth=0.25, linestyle="--", color="gray")
ax4.plot(date_list, semiempirical_low2, linewidth=0.1, linestyle=":", color="gray")
ax4.text(0.0125, 0.1, labels[3], transform=ax4.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')

for n in range(len(w_output)):
    ax4.plot([date_list[w_output[n]], date_list[w_output[n]]],
             [training_data[n] - err_mean, training_data[n] + err_mean],
             linewidth=0.8, color="k")
    ax4.plot([date_list[w_output[n]], date_list[w_output[n]]],
             [training_data[n] - 2.0 * err_mean, training_data[n] + 2.0 * err_mean],
             linewidth=0.0, marker="_", color="k")
ax4.plot(date_list[w_output], training_data, markersize=4., linewidth=0.0, marker="o", color="k", label="Surveys")

num_years = 10
event_counts = numpy.zeros(num_years, dtype=numpy.int64)
date_list_storms = []

def compute_event_counts(function_out2, w_time, ero_output, date_years, event_counts):
    ero_output_tmp = ero_output.copy()
    w1_tmp = numpy.where(ero_output[:,1]<par_09_mean)[0]
    ero_output_tmp[w1_tmp,1]=par_09_mean*1.0
    low_val = function_out2[-1]
    for n2 in range(len(function_out2)):
        idx = len(function_out2) - (n2 + 1)
        valu = function_out2[idx]
        if valu < low_val:
            low_val = valu
            ero_idx_start = max(idx - int(par_11_mean), 0)
            ero_idx_end = min(idx + int(par_12_mean), len(w_time))
            semi_idx_start = max(idx - int(par_11_mean), 0)
            semi_idx_end = min(idx + int(par_11_mean), len(function_out2))
            ero_min = numpy.min(ero_output[w_time[ero_idx_start:ero_idx_end], 1])
            semi_min = numpy.min(function_out2[semi_idx_start:semi_idx_end])
            if (ero_min < par_10_mean) and (valu <= semi_min):
                event_year = date_years[idx]
                year_index = event_year - 2009
                if 0 <= year_index < num_years:
                    date_list_storms.append([date_list[idx]])
                    ax4.plot([date_list[idx], date_list[idx]], [valu, valu],
                             marker="o", markersize=4., linewidth=0.0, color="red")
                    event_counts[year_index] += 1
    return event_counts

event_counts_vector = compute_event_counts(semiempirical_mean, w_time, ero_output, numpy.array([date.year for date in date_list], dtype=numpy.int64), event_counts*0)
ax4.plot([0,0],[-1000,-1000], marker="o", linewidth=0.0, color="red",
         label=f"Model predicted preserved\nerosion surfaces (n={str(int(numpy.sum(event_counts_vector)))})")

date_list_storms = numpy.array(date_list_storms)
date_list_storms_str = []
for idx in date_list_storms:
    for dt in idx:
        date_list_storms_str.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
date_list_file = wkspc + 'date_list_preserved_storms.csv'
numpy.savetxt(date_list_file, date_list_storms_str, fmt='%s', delimiter=',')

ax4.legend(loc=2, fontsize=6)
ax4.set_ylim(-50, 350)
ax4.grid()
ax4.set_xlim(d1, d2)
ax4.set_xlabel("Date", fontsize=6)
ax4.set_ylabel("Areal change of\nNB cross-section (m$^2$)", fontsize=6)
plt.tight_layout()
pltname = wkspc + "SemiEmpirical_Holgate.png"
plt.savefig(pltname, dpi=300)
plt.show()