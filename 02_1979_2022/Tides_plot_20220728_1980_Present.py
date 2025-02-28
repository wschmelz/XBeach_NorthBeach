import os
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import csv
import numpy
import random
import glob
import scipy
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.dates as mdates
from numpy import matrix
from numpy import genfromtxt
from numpy import linalg

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print(str(__file__))
print(time.strftime("%H:%M:%S"))

file_out = wkspc + "ero_output.csv"
tide_data = numpy.genfromtxt(file_out, delimiter=',')
tide_data[:,1] = tide_data[:,1] * -1.

year_tmp = 2009
month_tmp = 9
day_tmp = 15
hour_tmp = 0
time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)

tide_output_tmp = numpy.zeros((len(tide_data[:,0]),2))
tide_output_tmp[:,0] = tide_data[:,0] * 1.0
tide_output_tmp[:,1] = tide_data[:,1] * 1.0

tide_output_tmp2 = numpy.zeros((len(tide_data[:,0]),2))
tide_output_tmp3 = numpy.zeros((len(tide_data[:,0]),2))
date_list = []

for n2 in range(0,len(tide_data[:,0])):
    time_datum_stop = time_datum_start + datetime.timedelta(days = tide_data[n2,0])
    time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
    time_tmp_years = (time_datum_stop - time_datum_start).total_seconds() / (86400*365.25)
    date_list.append([time_datum_stop])
    tide_output_tmp[n2,0] = time_tmp
    tide_output_tmp[n2,1] = tide_data[n2,1]
    tide_output_tmp2[n2,0] = numpy.floor(time_tmp)
    tide_output_tmp2[n2,1] = tide_data[n2,1]
    tide_output_tmp3[n2,0] = numpy.floor(time_tmp_years-0.294)
    tide_output_tmp3[n2,1] = tide_data[n2,1]

w1 = numpy.where(numpy.isnan(tide_output_tmp[:,1])==False)[0]
slope, intercept, r, p, se = scipy.stats.linregress(tide_output_tmp[w1,0], tide_output_tmp[w1,1])
res = scipy.stats.linregress(tide_output_tmp[w1,0], tide_output_tmp[w1,1])
print(res)
print(slope)
print(se)
print("slope = " + str(res.slope), "intercept = " + str(res.intercept))

tinv = lambda p, df: abs(scipy.stats.t.ppf(p/2, df))
ts = tinv(0.05, len(w1)-2)
print("Error of slope: +/- ", ts*res.stderr)

var_95 = numpy.nanquantile(tide_output_tmp[:,1],0.975)
var_95 = 8.0

w1 = numpy.where(tide_output_tmp[:,1]>=var_95)[0]
days_list = numpy.unique(tide_output_tmp2[:,0])
daily_high = numpy.zeros((len(days_list),4))
date_list2 = []

for n in range(0,len(days_list)):
    w1 = numpy.where(tide_output_tmp2[:,0]==days_list[n])[0]
    w2 = numpy.where(numpy.isnan(tide_output_tmp2[w1,1])==False)[0]
    daily_high[n,2] = daily_high[n,2] * numpy.nan
    if len(w2) > 0:
        w3 = numpy.nanargmax(tide_output_tmp2[w1[w2],1])
        daily_high[n,0] = days_list[n]
        daily_high[n,1] = tide_output_tmp2[w1[w2[w3]],0]
        daily_high[n,2] = numpy.nanmax(tide_output_tmp2[w1[w2[w3]],1])
        daily_high[n,3] = tide_output_tmp3[w1[w2[w3]],0]
        time_datum_stop = time_datum_start + datetime.timedelta(days = tide_data[w1[w2[w3]],0])
        date_list2.append([time_datum_stop])
    else:
        date_list2.append([time_datum_stop])

threshold = var_95 * 1.0
w4 = numpy.where(daily_high[:,2]>=threshold)[0]
w5 = numpy.where(numpy.diff(daily_high[w4,0])>2.)[0]
events = numpy.split(w4, w5+1)
events_out = numpy.zeros((len(events),4))
date_list3 = []
print("")
print("Storm events: ")

for n in range(0,len(events)):
    indices = events[n]
    w6 = numpy.nanargmax(daily_high[indices,2])
    events_out[n,0] = daily_high[indices[w6],0]
    events_out[n,1] = daily_high[indices[w6],1]
    events_out[n,2] = daily_high[indices[w6],2]
    events_out[n,3] = daily_high[indices[w6],3]
    date_list3.append(date_list2[indices[w6]])
    print("Date: ", str(date_list2[indices[w6]][0])[0:11], "Max WL: ", round(daily_high[indices[w6],2],2), " ft.")
print("____")

mean_val1 = numpy.nanmean(events_out[:,2])
w1 = numpy.where(daily_high[:,2]>var_95)[0]
print(len(events))
print(len(events)/(len(days_list)*365.25))

w1 = numpy.flipud(numpy.argsort(events_out[:,2]))[49]
w2 = numpy.flipud(numpy.argsort(events_out[:,2]))[25]
w3 = numpy.flipud(numpy.argsort(events_out[:,2]))[6]
w4 = numpy.flipud(numpy.argsort(events_out[:,2]))[3]
print("biannual: ",events_out[w1,2])
print("yearly: ",events_out[w2,2])
print("5y: ",events_out[w3,2])
print("10y: ",events_out[w4,2])

fig = plt.figure(1,figsize=(6.5,6.5))
ax1 = plt.subplot(211)
ax1.plot(date_list, tide_output_tmp[:,1], linewidth=1., color="dodgerblue")
ax1.plot([date_list[0],date_list[-1]],[var_95,var_95], linewidth=2.5, color="red", label="Storm threshold - 8.0 m²")
new_y = (tide_output_tmp[:,0] * res.slope) + res.intercept
ax1.plot(date_list, new_y, linewidth=1.5, color="black")
ax1.plot(date_list3, events_out[:,2], linewidth=0., marker="x", color="blue", markersize=4, markeredgewidth=1.5, label="Storm event")
ax1.yaxis.set_minor_locator(MultipleLocator(1.))
ax1.yaxis.set_major_locator(MultipleLocator(5.))
ax1.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1,month=1,day=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.grid(which='minor', linewidth=.35, linestyle=":")
ax1.grid(which='major', linewidth=.5, linestyle="-")
ax1.set_xlim(date_list[0], date_list[-1])
ax1.set_ylim(0.0,50.5)
ax1.legend(fontsize=8,loc=2)
ax1.set_ylabel("Model predicted erosion (m² over 48h)", fontsize=10)
ax1.set_xticklabels([])
ax1.tick_params(axis='y', labelsize=8)
ax1.tick_params(axis='x', labelsize=8)
at = AnchoredText('A', prop=dict(size=8,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

ax2 = plt.subplot(212)
years_unique = numpy.unique(events_out[:,3])
output_frequency = numpy.zeros((45,2))
date_list4 = []

for n in range(0,45):
    output_frequency[n,0] = int(int(numpy.min(tide_output_tmp3[:,0]))+n)
    w11 = numpy.where(years_unique==int(int(numpy.min(tide_output_tmp3[:,0]))+n))[0]
    if len(w11) > 0:
        w1 = numpy.where(events_out[:,3]==int(int(numpy.min(tide_output_tmp3[:,0]))+n))[0]
        if len(w1) > 0:
            output_frequency[n,1] = len(w1)
    survey_date1 = datetime.datetime(int(n + 1979), 7, 2)
    date_list4.append(survey_date1)

strat_storms = numpy.genfromtxt(wkspc + "Stratigraphic_events.csv", delimiter=",")
w2 = numpy.where(strat_storms)
ax2.plot(date_list4, output_frequency[:,1], linewidth=0., marker="o", color="blue", markersize=4, markeredgewidth=0, label="Storm events")

res = scipy.stats.linregress(output_frequency[0:-1,0] - numpy.mean(output_frequency[0:-1,0]), output_frequency[0:-1,1] - numpy.mean(output_frequency[0:-1,1]))

print(res)
print(slope)
print(se)
print("slope = " + str(res.slope), "intercept = " + str(res.intercept))
p = res.pvalue
ts = tinv(0.05, len(output_frequency[:,0])-2)

mc_iters = 50000
mc_output = numpy.zeros((mc_iters,len(output_frequency[0:-1,0])))

for n in range(0,mc_iters):
    slope_random = numpy.random.normal(res.slope,res.stderr)
    intercept_random = numpy.random.normal(res.intercept,res.intercept_stderr)
    mc_output[n,:] = (slope_random*(output_frequency[0:-1,0] - numpy.mean(output_frequency[0:-1,0]))) + intercept_random

low_2x = numpy.quantile(mc_output,0.025,axis=0) + numpy.mean(output_frequency[0:-1,1])
low = numpy.quantile(mc_output,0.16,axis=0) + numpy.mean(output_frequency[0:-1,1])
median = numpy.quantile(mc_output,0.5,axis=0) + numpy.mean(output_frequency[0:-1,1])
high = numpy.quantile(mc_output,0.84,axis=0) + numpy.mean(output_frequency[0:-1,1])
high_2x = numpy.quantile(mc_output,0.975,axis=0) + numpy.mean(output_frequency[0:-1,1])

new_x = output_frequency[0:-1,0] * 1.0
ax2.plot(date_list4[0:-1], median, linestyle="-", linewidth=1.0, color="b", label="Regression of\nstorm events")
ax2.plot(date_list4[0:-1], low, linestyle="--", linewidth=0.5, color="b")
ax2.plot(date_list4[0:-1], high, linestyle="--", linewidth=0.5, color="b")
ax2.plot(date_list4[0:-1], low_2x, linestyle=":", linewidth=0.5, color="b")
ax2.plot(date_list4[0:-1], high_2x, linestyle=":", linewidth=0.5, color="b")

ax2.plot(date_list4[30:40], strat_storms[:,1], linewidth=0., marker="x", color="r", markersize=4, label="Stratigraphic events")
ax2.plot([0,0],[0,0], linewidth=0.5, color="k", label="Rate of shoreline\npos. change at SH8")

shoreline_pos = numpy.genfromtxt(wkspc + "shoreline_growth.csv", delimiter=",")
time_datum_start = datetime.datetime(int(shoreline_pos[0,0]), int(shoreline_pos[0,1]), int(shoreline_pos[0,2]))
storm_output_tmp = numpy.zeros((len(shoreline_pos[:,0])*2-2,2))
date_list_shore = []

for n in range(1,len(shoreline_pos[:,0])):
    year_tmpm1 = int(shoreline_pos[n-1,0])
    month_tmpm1 = int(shoreline_pos[n-1,1])
    day_tmpm1 = int(shoreline_pos[n-1,2])
    shore_posm1 = shoreline_pos[n-1,3]
    time_datum_stopm1 = datetime.datetime(year_tmpm1, month_tmpm1, day_tmpm1)
    time_tmp_yearsm1 = (time_datum_stopm1 - time_datum_start).total_seconds() / (86400*365.25)
    time_datum_stopm2 = datetime.datetime(year_tmpm1, month_tmpm1, int(day_tmpm1+1))
    date_list_shore.append(time_datum_stopm2)
    year_tmp = int(shoreline_pos[n,0])
    month_tmp = int(shoreline_pos[n,1])
    day_tmp = int(shoreline_pos[n,2])
    shore_pos = shoreline_pos[n,3]
    time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp)
    time_tmp_years = (time_datum_stop - time_datum_start).total_seconds() / (86400*365.25)
    date_list_shore.append(time_datum_stop)
    shore_growth = (shore_pos - shore_posm1)/(time_tmp_years - time_tmp_yearsm1)
    storm_output_tmp[int((n-1)*2),0] = time_tmp_yearsm1
    storm_output_tmp[int((n-1)*2)+1,0] = time_tmp_years
    storm_output_tmp[int((n-1)*2),1] = shore_growth
    storm_output_tmp[int((n-1)*2)+1,1] = shore_growth

ax2.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))
ax2.xaxis.set_minor_locator(mdates.YearLocator(1,month=1,day=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.grid(which='minor', linewidth=.35, linestyle=":")
ax2.grid(which='major', linewidth=.5, linestyle="-")
ax2.set_xlim(date_list[0], date_list[-1])
ax2.set_ylim(0,8.5)

ax3 = ax2.twinx()
ax3.plot(date_list_shore, storm_output_tmp[:,1], linewidth=0.5, color="k")
ax3.set_ylabel("Rate of change (m/y)", fontsize=10)

ax2.legend(fontsize=8, loc=2)
ax2.tick_params(axis='both', labelsize=8)
ax2.set_xlabel("Time (Years)", fontsize=10)
ax2.set_ylabel("Number of events", fontsize=10)
at = AnchoredText('B', prop=dict(size=8,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)

pltname = wkspc + "SH_Storms_1980_present.png"
plt.tight_layout()
plt.savefig(pltname)
plt.show()