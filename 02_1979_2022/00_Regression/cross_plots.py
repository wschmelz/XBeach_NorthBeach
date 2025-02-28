import os
import sys
import datetime
import shutil
import time
import glob
import numpy
from numpy import matrix, genfromtxt, linalg
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import requests
import scipy
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy import stats

'''
url = 'https://raw.githubusercontent.com/wschmelz/GeologicalModeling/main/Scripts/m_avg.py'
m_avg_py = requests.get(url)  
with open('m_avg_py.py', 'w') as f:
    f.write(m_avg_py.text)
'''

import m_avg_py

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
data_wkspc = wkspc + "00_DataPrep/"

def convert_seconds_to_real_time(seconds, start_time):

    return [start_time + datetime.timedelta(days=s) for s in seconds]

def lr_one_vars(x, z_obs, x_missing):

    ones_train = numpy.ones(len(x))
    X_train = numpy.column_stack((ones_train, x))
    y_train = z_obs
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_train, y_train)
    R_squared = lr_model.score(X_train, y_train)
    print(f"Coefficient of Determination (R^2): {R_squared:.4f}")
    y_pred_train = lr_model.predict(X_train)
    r, _ = pearsonr(y_train, y_pred_train)
    print(f"Pearson Correlation Coefficient (r): {r:.4f}")
    ones_missing = numpy.ones(len(x_missing))
    X_missing = numpy.column_stack((ones_missing, x_missing))
    z_predicted = lr_model.predict(X_missing)
    z_predicted_train = y_pred_train
    return z_predicted, z_predicted_train

def lr_two_vars(x, y, z_obs, x_missing, y_missing):

    ones_train = numpy.ones(len(x))
    X_train = numpy.column_stack((ones_train, x, y))
    y_train = z_obs
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_train, y_train)
    R_squared = lr_model.score(X_train, y_train)
    print(f"Coefficient of Determination (R^2): {R_squared:.4f}")
    y_pred_train = lr_model.predict(X_train)
    r, _ = pearsonr(y_train, y_pred_train)
    print(f"Pearson Correlation Coefficient (r): {r:.4f}")
    ones_missing = numpy.ones(len(x_missing))
    X_missing = numpy.column_stack((ones_missing, x_missing, y_missing))
    z_predicted = lr_model.predict(X_missing)
    z_predicted_train = y_pred_train
    return z_predicted, z_predicted_train

def infill_missing_data_lr_three_vars(t, x, y, z_obs, t_missing, x_missing, y_missing):

    ones_train = numpy.ones(len(t))
    X_train_lr = numpy.column_stack((ones_train, t, x, y))
    y_train = z_obs
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_train_lr, y_train)
    R_squared_lr = lr_model.score(X_train_lr, y_train)
    print(f"Linear Regression Coefficient of Determination (R^2): {R_squared_lr:.4f}")
    z_predicted_train_lr = lr_model.predict(X_train_lr)
    r_lr, _ = pearsonr(y_train, z_predicted_train_lr)
    print(f"Linear Regression Pearson Correlation Coefficient (r): {r_lr:.4f}")
    ones_missing = numpy.ones(len(t_missing))
    X_missing_lr = numpy.column_stack((ones_missing, t_missing, x_missing, y_missing))
    z_predicted_lr = lr_model.predict(X_missing_lr)
    residuals = y_train - z_predicted_train_lr
    X_train_rf = numpy.column_stack((t, x, y))
    rf_model = RandomForestRegressor(n_estimators=250, random_state=0)
    rf_model.fit(X_train_rf, residuals)
    X_missing_rf = numpy.column_stack((t_missing, x_missing, y_missing))
    residuals_predicted = rf_model.predict(X_missing_rf)
    residuals_predicted_train = rf_model.predict(X_train_rf)
    z_final_predicted = z_predicted_lr + residuals_predicted
    z_final_predicted_train = z_predicted_train_lr + residuals_predicted_train
    R_squared_final = r2_score(y_train, z_final_predicted_train)
    print(f"Final Model Coefficient of Determination (R^2): {R_squared_final:.4f}")
    r_final, _ = pearsonr(y_train, z_final_predicted_train)
    print(f"Final Model Pearson Correlation Coefficient (r): {r_final:.4f}")
    return z_final_predicted, z_final_predicted_train
	
def infill_missing_data_lr_two_vars(t, x, y, z_obs, t_missing, x_missing, y_missing):

    ones_train = numpy.ones(len(t))
    X_train_lr = numpy.column_stack((ones_train, x, y))
    y_train = z_obs
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_train_lr, y_train)
    R_squared_lr = lr_model.score(X_train_lr, y_train)
    print(f"Linear Regression Coefficient of Determination (R^2): {R_squared_lr:.4f}")
    z_predicted_train_lr = lr_model.predict(X_train_lr)
    r_lr, _ = pearsonr(y_train, z_predicted_train_lr)
    print(f"Linear Regression Pearson Correlation Coefficient (r): {r_lr:.4f}")
    ones_missing = numpy.ones(len(t_missing))
    X_missing_lr = numpy.column_stack((ones_missing, x_missing, y_missing))
    z_predicted_lr = lr_model.predict(X_missing_lr)
    residuals = y_train - z_predicted_train_lr
    X_train_rf = numpy.column_stack((x, y))
    rf_model = RandomForestRegressor(n_estimators=250, random_state=0)
    rf_model.fit(X_train_rf, residuals)
    X_missing_rf = numpy.column_stack((x_missing, y_missing))
    residuals_predicted = rf_model.predict(X_missing_rf)
    residuals_predicted_train = rf_model.predict(X_train_rf)
    z_final_predicted = z_predicted_lr + residuals_predicted
    z_final_predicted_train = z_predicted_train_lr + residuals_predicted_train
    R_squared_final = r2_score(y_train, z_final_predicted_train)
    print(f"Final Model Coefficient of Determination (R^2): {R_squared_final:.4f}")
    r_final, _ = pearsonr(y_train, z_final_predicted_train)
    print(f"Final Model Pearson Correlation Coefficient (r): {r_final:.4f}")
    return z_final_predicted, z_final_predicted_train

geomorph_files = glob.glob(wkspc + "output_geomorph_48h_*.csv")
today_values = [file.split('48h_')[-1].split('.')[0] for file in geomorph_files]
all_output_wave = []
all_output_tide = []
all_output_runup = []
all_differenced_geomorph = []
all_max_wave = []
all_max_tide = []
all_mean_runup = []
all_total_geomorph_change = []
fiv_hund_list =[]
w_500 = []

for today in today_values:
    val_id = today.split('_')[0]
    if int(val_id) >=500:
        fiv_hund_list.append(val_id,)
        fiv_hund_arr = numpy.array(fiv_hund_list)
        w_500 = numpy.where(fiv_hund_arr==val_id)[0]
    if (int(val_id) <500) | ((int(val_id) >=500)&(len(w_500) < 2)):
        output_wave = numpy.genfromtxt(wkspc + 'output_wave_48h_' + str(today) + '.csv', delimiter=',')
        output_tide = numpy.genfromtxt(wkspc + 'output_tide_48h_' + str(today) + '.csv', delimiter=',')
        output_runup = numpy.genfromtxt(wkspc + 'output_runup_48h_' + str(today) + '.csv', delimiter=',')
        output_geomorph = numpy.genfromtxt(wkspc + 'output_geomorph_48h_' + str(today) + '.csv', delimiter=',')
        differenced_geomorph = numpy.diff(output_geomorph)
        total_geomorph_change = output_geomorph[-1] - output_geomorph[0]
        all_output_wave.extend(output_wave[1:])
        all_output_tide.extend(output_tide[1:])
        all_output_runup.extend(output_runup[1:])
        all_differenced_geomorph.extend(differenced_geomorph)
        all_max_wave.append(numpy.mean(output_wave))
        all_max_tide.append(numpy.mean(output_tide))
        all_mean_runup.append(numpy.mean(output_runup))
        all_total_geomorph_change.append(total_geomorph_change)

all_output_wave = numpy.array(all_output_wave)
all_output_tide = numpy.array(all_output_tide)
all_output_runup = numpy.array(all_output_runup)
all_differenced_geomorph = numpy.array(all_differenced_geomorph)
all_max_wave = numpy.array(all_max_wave)
all_max_tide = numpy.array(all_max_tide)
all_mean_runup = numpy.array(all_mean_runup)
all_total_geomorph_change = numpy.array(all_total_geomorph_change)

fig, axs = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle(f'Modeled hours: {int(len(today_values)*48)}, 48h model runs: {len(today_values)}', fontsize= 18)

def plot_crossplot(ax, x, y, z, xlabel, ylabel):

    sort = numpy.argsort(numpy.absolute(z))
    x = x[sort]
    y = y[sort]
    z = z[sort]
    sc = ax.scatter(x, y, c=z, s=25, alpha=0.95, cmap='RdBu', vmin=-1, vmax=1,
                    edgecolor='lightgray', linewidth=0.25)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True)
    return sc

def plot_crossplot2(ax, x, y, z, xlabel, ylabel):

    sc = ax.scatter(x, y, c=z, s=25, alpha=0.95, cmap='RdBu', vmin=-7.5, vmax=7.5,
                    edgecolor='lightgray', linewidth=0.25)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True)
    return sc

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
plot_crossplot(axs[0, 0], all_output_wave, all_differenced_geomorph, all_differenced_geomorph,
               'SWH (m)', 'Areal change (m²/h)')
plot_crossplot(axs[0, 1], all_output_tide, all_differenced_geomorph, all_differenced_geomorph,
               'WL (m. rel. NAVD88)', 'Areal change (m²/h)')
plot_crossplot(axs[0, 2], all_output_runup, all_differenced_geomorph, all_differenced_geomorph,
               'Runup (m. rel. NAVD88)', 'Areal change (m²/h)')
plot_crossplot(axs[0, 3], all_output_tide, all_output_wave, all_differenced_geomorph,
               'WL (m. rel. NAVD88)', 'SWH (m)')
plot_crossplot2(axs[1, 0], all_max_wave, all_total_geomorph_change, all_total_geomorph_change,
                '48h avg. SWH (m)', 'Areal change (m² over 48h)')
plot_crossplot2(axs[1, 1], all_max_tide, all_total_geomorph_change, all_total_geomorph_change,
                '48h avg. WL (m. rel. NAVD88)', 'Areal change (m² over 48h)')
plot_crossplot2(axs[1, 2], all_mean_runup, all_total_geomorph_change, all_total_geomorph_change,
                '48h avg. runup (m. rel. NAVD88)', 'Areal change (m² over 48h)')
plot_crossplot2(axs[1, 3], all_max_tide, all_max_wave, all_total_geomorph_change,
                '48h avg. WL (m. rel. NAVD88)', '48h avg. SWH (m)')

for i, ax in enumerate(axs.flat):
    if (i == 3) or (i == 7):
        ax.text(0.05, 0.95, labels[i], transform=ax.transAxes, fontsize=18, fontweight='bold',
                va='top', ha='left')
    else:
        ax.text(0.05, 0.05, labels[i], transform=ax.transAxes, fontsize=18, fontweight='bold',
                va='bottom', ha='left')

plt.tight_layout()
today1 = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = "cross_plot_" + str(today1) + ".png"
pltname = wkspc + filename
plt.savefig(pltname, dpi=300)
plt.close()

wave_file = wkspc + "wave_output_1979_2022.csv"
tide_file = wkspc + "tide_output_1979_2022.csv"
wave_output = numpy.genfromtxt(wave_file,delimiter=',')
tide_output = numpy.genfromtxt(tide_file,delimiter=',')
tide_output[:,1] = (tide_output[:,1] - 2.82)/3.28084
start_time = datetime.datetime(2009, 9, 15, 0)
tide_time_real = convert_seconds_to_real_time(tide_output[:, 0], start_time)
wave_time_real = convert_seconds_to_real_time(wave_output[:, 0], start_time)
w_realvals = numpy.where((numpy.isnan(tide_output[:,1])==False)&(numpy.isnan(wave_output[:,1])==False))[0]
wave_time_real2 = convert_seconds_to_real_time(wave_output[w_realvals, 0], start_time)
z_predicted, z_predicted_train = lr_two_vars(all_output_tide, all_output_wave, all_output_runup,
                                                                 tide_output[w_realvals,1], wave_output[w_realvals,1])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 8))
ax1.plot(tide_time_real, tide_output[:, 1], linewidth=1.0, color='dodgerblue', label='Original Tides')
ax1.set_xlabel('Time')
ax1.set_ylabel('Tide Level')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.grid(True, which='both')

ax2.plot(wave_time_real, wave_output[:, 1], linewidth=1.0, color='red', label='Original Waves')
ax2.set_xlabel('Time')
ax2.set_ylabel('Wave Level')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.grid(True, which='both')

w_max = numpy.where(z_predicted>6.0)[0]
z_predicted[w_max] = 6.0
ax3.plot(wave_time_real2, z_predicted, linewidth=1.0, color='k', label='Runup')
ax3.set_xlabel('Time')
ax3.set_ylabel('Runup')
ax3.legend()
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_minor_locator(mdates.MonthLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.grid(True, which='both')

w1 = numpy.where((all_output_runup>1.25)&(all_differenced_geomorph>0.25))[0]
print(w1)
all_output_tide = numpy.delete(all_output_tide,w1)
all_output_wave = numpy.delete(all_output_wave,w1)
all_output_runup = numpy.delete(all_output_runup,w1)
all_differenced_geomorph = numpy.delete(all_differenced_geomorph,w1)

span = 1.
tide_MA = m_avg_py.m_avg(tide_output[w_realvals, 0],tide_output[w_realvals, 0],tide_output[w_realvals, 1],span)
wave_MA = m_avg_py.m_avg(wave_output[w_realvals, 0],wave_output[w_realvals, 0],wave_output[w_realvals, 1],span)
runup_MA = m_avg_py.m_avg(wave_output[w_realvals, 0],wave_output[w_realvals, 0],z_predicted,span)
z_predicted_ero, z_predicted_train_ero = infill_missing_data_lr_three_vars(all_max_tide, all_max_wave, all_mean_runup,
                                                                           all_total_geomorph_change,
                                                                           tide_MA, wave_MA, runup_MA)
ero_output = numpy.zeros((len(wave_output[:,0]),2)) * numpy.nan
ero_output[:,0] = wave_output[:,0]
ero_output[w_realvals,1] = z_predicted_ero
w1 = numpy.where(numpy.isnan(ero_output[:,1]))[0]
mean = numpy.mean(z_predicted_ero)
std = numpy.std(z_predicted_ero)/7.5
ero_output[:,1] = numpy.random.normal(loc=mean,scale=std,size=len(ero_output[:,1]))
ero_output[w_realvals,1] = z_predicted_ero
ero_thresh = -8.
w_close5 = numpy.where(numpy.absolute(ero_output[w_realvals,1]-ero_thresh)<0.1)[0]
tide_equiv = numpy.mean(tide_output[w_realvals[w_close5],1])
print((tide_equiv*3.28084)+ 2.82)
wave_equiv = numpy.mean(wave_output[w_realvals[w_close5],1])
print(wave_equiv)
ero_file = wkspc + 'ero_output.csv'
numpy.savetxt(ero_file, ero_output, delimiter=',')

ax4.plot(tide_time_real, ero_output[:,1], linewidth=1.0, color='gray', label='erosion')
ax4.set_xlabel('Time')
ax4.set_ylabel('Erosion')
ax4.legend()
ax4.xaxis.set_major_locator(mdates.YearLocator())
ax4.xaxis.set_minor_locator(mdates.MonthLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4.grid(True, which='both')

plt.tight_layout()
plt.savefig('runup_ero_pred.png')
plt.close()

print(time.strftime("%H:%M:%S"))
print(numpy.shape(z_predicted_ero))

tide_data = numpy.zeros((len(z_predicted_ero),5))
for n in range(0,len(z_predicted_ero)):
    dt1 = wave_time_real2[n]
    tide_data[n,0] = dt1.year
    tide_data[n,1] = dt1.month
    tide_data[n,2] = dt1.day
    tide_data[n,3] = dt1.hour
tide_data[:,4] = z_predicted_ero

year_tmp = int(tide_data[0,0])
month_tmp = int(tide_data[0,1])
day_tmp = int(tide_data[0,2])
hour_tmp = int(tide_data[0,3])
tide_output_tmp = numpy.zeros((len(tide_data[:,0]),2))
tide_output_tmp2 = numpy.zeros((len(tide_data[:,0]),2))
tide_output_tmp3 = numpy.zeros((len(tide_data[:,0]),2))
date_list = []

for n2 in range(0,len(tide_data[:,0])):    
    year_tmp = int(tide_data[n2,0])
    month_tmp = int(tide_data[n2,1])
    day_tmp = int(tide_data[n2,2])
    hour_tmp = int(tide_data[n2,3])
    time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
    time_tmp = (time_datum_stop - start_time).total_seconds() / (3600.0*24.)
    time_tmp_years = (time_datum_stop - start_time).total_seconds() / (86400*365.25)
    date_list.append(time_datum_stop,)
    tide_output_tmp[n2,0] = time_tmp
    tide_output_tmp[n2,1] = tide_data[n2,4]    
    tide_output_tmp2[n2,0] = numpy.floor(time_tmp)
    tide_output_tmp2[n2,1] = tide_data[n2,4]
    tide_output_tmp3[n2,0] = numpy.floor(time_tmp_years)
    tide_output_tmp3[n2,1] = tide_data[n2,4]

var_95 = numpy.nanquantile(tide_output_tmp[:,1],0.005)
var_95 = (ero_thresh + 0.0 ) * 1.0
threshold = numpy.nanquantile(tide_output_tmp[:,1],0.005)
threshold =(ero_thresh + 0.0 )* 1.0
event_idx = numpy.where(tide_output_tmp[:, 1] < threshold)[0]
events = []
storm_list = []
current_event = [event_idx[0]]

for idx in event_idx[1:]:
    time_gap = tide_output_tmp[idx, 0] - tide_output_tmp[current_event[-1], 0]
    if time_gap <= 2.0:
        current_event.append(idx)
    else:
        events.append(current_event)
        current_event = [idx]
events.append(current_event)

fig = plt.figure(1,figsize=(12,6.5))

ax1 = plt.subplot(111)
ax1.plot(date_list, tide_output_tmp[:,1], linewidth=1., color="gray")
ax1.plot([date_list[0],date_list[-1]],[var_95,var_95], linewidth=2.5, color="red",
         label=f"Major storm threshold: {round(var_95,1)} m")
		 
survey_date1 = datetime.datetime(2009, 9, 15)
survey_date_orig = datetime.datetime(2009, 1, 1)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="peru")
ax1.text(survey_date1,-16.5,"09/15/2009", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2010, 10, 7)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="darkorange")
ax1.text(survey_date1,-16.5,"10/7/2010", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2011, 11, 12)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="lawngreen")
ax1.text(survey_date1,-16.5,"11/12/2011", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2012, 9, 26)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="darkgreen")
ax1.text(survey_date1,-16.5,"9/26/2012", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2013, 9, 13)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="aquamarine")
ax1.text(survey_date1,-16.5,"9/13/2013", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2014, 9, 10)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="darkviolet")
ax1.text(survey_date1,-16.5,"9/10/2014", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2015, 4, 22)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="pink")
ax1.text(survey_date1,-16.5,"4/22/2015", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2016, 9, 26)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="olive")
ax1.text(survey_date1,-16.5,"9/26/2016", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2017, 9, 18)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="lime")
ax1.text(survey_date1,-16.5,"9/18/2017", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)
survey_date1 = datetime.datetime(2018, 9, 27)
survey_date_final = datetime.datetime(2019, 1, 1)
ax1.plot([survey_date1,survey_date1],[-16,100], linewidth=2.5, color="paleturquoise")
ax1.text(survey_date1,-16.5,"9/27/2018", horizontalalignment="right", weight=550, rotation="vertical", fontsize=8)

count_storms = 0
tide_max = []
wave_max = []
storm_max = []

for i, event in enumerate(events):
    event_levels = tide_output_tmp[event, 1]
    max_level = numpy.min(event_levels)
    max_idx_within_event = numpy.argmin(event_levels)
    max_index = event[max_idx_within_event]
    event_max_time = tide_output_tmp[max_index, 0]
    event_max_datelist = date_list[max_index]
    w1 = numpy.array(event,dtype=int)
    date_list_array = numpy.array(date_list)
    tide_wave_event = numpy.where(numpy.absolute(tide_output_tmp[max_index, 0] - tide_output[:,0])<=1)[0]
    tides_m = numpy.nanmax(tide_output[tide_wave_event,1])
    waves_m = numpy.nanmax(wave_output[tide_wave_event,1]) 
    tide_max.append(tides_m,)
    wave_max.append(waves_m,)
    storm_max.append(max_level,)
    event_datelist = date_list_array[w1]
    if (event_datelist[0].year <= 2023) and (event_datelist[0].year >= 1979):
        count_storms+=1
        print(f"Event {i+1}:")
        print(f"  Max Erosion Level: {max_level} meters at {event_max_datelist}")
        print(f"  Max Wave Level: {waves_m} meters")
        print(f"  Max Tide Level: {tides_m} meters")

tide_max = numpy.array(tide_max)
wave_max = numpy.array(wave_max)
storm_max = numpy.array(storm_max)
w1 = numpy.where((numpy.isnan(wave_max)==False)&(numpy.isnan(tide_max)==False))[0]
w2 = numpy.where((wave_max[w1]>0.0)&(tide_max[w1]>0.0))[0]
wave_pred,wave_pred2 = lr_one_vars(storm_max[w1[w2]], wave_max[w1[w2]], numpy.array([ero_thresh,ero_thresh]))
tide_pred,tide_pred2 = lr_one_vars(storm_max[w1[w2]], tide_max[w1[w2]], numpy.array([ero_thresh,ero_thresh]))
print("wave1",wave_pred)
print("tide1",(tide_pred *(1/.3048))+ 2.82)
wave_pred3,wave_pred4 = lr_two_vars(storm_max[w1[w2]], tide_max[w1[w2]], wave_max[w1[w2]],
                                    numpy.array([ero_thresh,ero_thresh]), numpy.array([tide_pred[0],tide_pred[0]]))
tide_pred3,tide_pred4 = lr_two_vars(storm_max[w1[w2]], wave_max[w1[w2]], tide_max[w1[w2]],
                                    numpy.array([ero_thresh,ero_thresh]), numpy.array([wave_pred[0],wave_pred[0]]))
print("wave2",wave_pred3)
print("tide2",(tide_pred3*(1/.3048))+ 2.82)

for i, event in enumerate(events):
    event_levels = tide_output_tmp[event, 1]
    max_level = numpy.min(event_levels)
    max_idx_within_event = numpy.argmin(event_levels)
    max_index = event[max_idx_within_event]
    event_max_time = tide_output_tmp[max_index, 0]
    event_max_datelist = date_list[max_index]
    w1 = numpy.array(event,dtype=int)
    date_list_array = numpy.array(date_list)
    event_datelist = date_list_array[w1]
    if (event_datelist[0].year <= 2023):
        if i == 0:
            ax1.plot(event_max_datelist, max_level, linewidth=0., marker="x", color="blue",
                     markersize=9, markeredgewidth=2,
                     label=f"Model predicted\nerosion event (n={str(int(count_storms))})")
        else:
            ax1.plot(event_max_datelist, max_level, linewidth=0., marker="x", color="blue",
                     markersize=9, markeredgewidth=2)
        ax1.plot(event_datelist, event_levels, linewidth=0.5, color="blue")
        print(f"Event {i+1}:")
        print(f"  Max Water Level: {max_level} meters at {event_max_datelist}")

date_list_storms_file = wkspc + 'date_list_preserved_storms.csv'
with open(date_list_storms_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]
date_list_preserved_storms = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                          for date_str in date_list_str])

count = 0
for n in range(0,len(date_list_preserved_storms)):
    date = date_list_preserved_storms[n]
    if date.year<=2023:
        if n ==0:
            ax1.plot([date,date],[-14.,100], linewidth=0.5, linestyle=":", color="r")
            ax1.plot(date,-14., linewidth=0., marker="x", color="red", markersize=9)
        if n !=0:
            ax1.plot([date,date],[-14.,100], linewidth=0.5, linestyle=":", color="r")
            ax1.plot(date,-14., linewidth=0., marker="x", color="red", markersize=9)
        count += 1
ax1.plot(date,-14., linewidth=0., marker="x", color="red", markersize=9,
         label=f"Model predicted preserved\nerosion surfaces (n={count})")

print(f"Total events included: {len(events)}")
ax1.xaxis.set_major_locator(mdates.YearLocator(1,month=1,day=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.grid(which='major', linewidth=0.5, linestyle='-')
ax1.grid(which='minor', linewidth=0.35, linestyle=':')
survey_date_orig = datetime.datetime(1979, 1, 1)
survey_date_final = datetime.datetime(2023, 1, 1)
ax1.set_xlim(survey_date_orig, survey_date_final)
ax1.set_ylim(0., -25.)
ax1.legend(loc=2)
ax1.set_ylabel("Model predicted erosion (m² over 48h)", fontsize=18)
ax1.set_xlabel("Time (Years)", fontsize=18)
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
pltname = wkspc + "SH_Storms_monthly_ws.png"
plt.tight_layout()
plt.savefig(pltname)