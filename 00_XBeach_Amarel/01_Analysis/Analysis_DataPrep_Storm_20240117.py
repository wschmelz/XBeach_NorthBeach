import sys
import os
import time
import numpy
import glob
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy
from scipy import stats
import loess2D
import cmep_xbeach as xb

print(sys.version)

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))
print (time.strftime("%H:%M:%S"))

task_id = int(sys.argv[1]) - 500

buoy_num = 44065
station_num = 8531680

tide_dir = wkspc + f"01_Station{station_num}/"
wave_dir = wkspc + f"00_Buoy{buoy_num}/"
vol_wkspc = wkspc + '02_TopoData/'

tide_file = wkspc + 'tide_output.csv'
tide_output = numpy.genfromtxt(tide_file, delimiter=',')

w1 = numpy.where(numpy.isnan(tide_output[:,1]))[0]
w2 = numpy.where(numpy.isnan(tide_output[:,1])==False)[0]
tide_output[w1,1] = numpy.nanmean(tide_output[w2,1]) + numpy.random.normal(loc=0,scale=numpy.nanstd(tide_output[w2,1]),size=len(w1))

tide_output[:,1] = (tide_output[:,1] - 2.82)/3.28084

wave_file = wkspc + 'wave_output.csv'
wave_output = numpy.genfromtxt(wave_file, delimiter=',')

beach_profile_file = wkspc + 'X_beach_SH8.csv'
beach_profile_tmp = numpy.genfromtxt(beach_profile_file, delimiter=',')

w1 = numpy.where(beach_profile_tmp[:,1]>=-10.1)[0]

beach_dists = numpy.arange(0,numpy.max(beach_profile_tmp[w1,0]),2.5)

beach_profile = numpy.zeros((len(beach_dists),3))
beach_profile[:,0] = beach_dists - numpy.max(beach_dists)
beach_profile[:,1] = (beach_dists - numpy.max(beach_dists))*0.0
beach_profile[:,2] = loess2D.loess_int(beach_dists,beach_dists*0.0,beach_profile_tmp[:,0],beach_profile_tmp[:,0]*0.0,beach_profile_tmp[:,1],3,25.,25.)[2]

beach_profile = numpy.flipud(beach_profile)

numpy.savetxt('beach_profile.csv', beach_profile, delimiter=',', fmt='%.3f')

date_list_file = wkspc + 'date_list.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]

date_list = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

year_tmp = 2020
month_tmp = 10
day_tmp = 17
hour_tmp = 0

time_datum_start = date_list[0]

time_datum_last_data = date_list[-1]

time_datum_first_data = date_list[0]

print (time_datum_start)
print (time_datum_last_data)
print (time_datum_first_data)

time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)

w_time = numpy.where((tide_output[:, 0] >= 0) & (tide_output[:, 0] <= time_tmp_last))[0]

dt = numpy.mean(tide_output[1:,0] - tide_output[0:-1,0])

threshold = 1.2
event_idx = numpy.where(tide_output[w_time, 1] > threshold)[0]

events = []
storm_list = []
current_event = [event_idx[0]]

for idx in event_idx[1:]:
	time_gap = tide_output[w_time[idx], 0] - tide_output[w_time[current_event[-1]], 0]
	if time_gap <= 2.0:
		current_event.append(idx)
	else:
		events.append(current_event)
		current_event = [idx]

events.append(current_event)

for i, event in enumerate(events):
	event_levels = tide_output[w_time[event], 1]
	max_level = numpy.max(event_levels)
	max_idx_within_event = numpy.argmax(event_levels)
	max_index = event[max_idx_within_event]
	event_time = tide_output[w_time[max_index], 0]

	total_points = 48
	half_points = total_points // 2
	start_index = max_index - half_points + 6
	end_index = max_index + half_points + 6

	if start_index < 0 or end_index > len(tide_output) - 1:
		print(f"Event {i+1} excluded due to boundary conditions.")
		continue

	w_seq = numpy.arange(start_index, end_index)

	go = numpy.sum(wave_output[w_seq, 3])

	if go >= 1:
		print(f"Event {i+1} excluded because go >= 1 (go={go}).")
		continue

	storm_list.append(w_seq)

	print(f"Event {i+1}:")
	print(f"  Max Water Level: {max_level} meters at time {event_time} days from start")
	print(f"  Data points from index {start_index} to {end_index}")
	print(f"Total events included: {len(storm_list)}")

print(storm_list[task_id])

w_seq = storm_list[task_id]

jnswp_wave_file = numpy.zeros((48,7))
WL_file = numpy.zeros((48,2))
n = numpy.arange(0.,48.,1.)

jnswp_wave_file[:,0] = wave_output[w_time[w_seq],1]
jnswp_wave_file[:,1] = wave_output[w_time[w_seq],2]

WL_file[:,0] = n * 3600.
WL_file[:,1] = tide_output[w_time[w_seq],1]

jnswp_wave_file[:,2] = jnswp_wave_file[:,2] + 270.
jnswp_wave_file[:,3] = jnswp_wave_file[:,3] + 3.3
jnswp_wave_file[:,4] = jnswp_wave_file[:,4] + 10.
jnswp_wave_file[:,5] = jnswp_wave_file[:,5] + 3600.
jnswp_wave_file[:,6] = jnswp_wave_file[:,6] + 1.

MaxWL = numpy.max(WL_file[:,1])
MinWL = numpy.min(WL_file[:,1])

w1 = numpy.where(jnswp_wave_file[:,0]<0.05)[0]
jnswp_wave_file[w1,0] = 0.05
numpy.savetxt('tide.txt', WL_file, delimiter=' ', fmt='%.3f')
numpy.savetxt('jonswap_table.txt', jnswp_wave_file, delimiter=' ', fmt='%.3f')

ProfileFile = 'beach_profile.csv' 
ErosionEnabled = True
StructureElevation = None
CliffedBeach = False
Dsf = None 
Slope = None

XBeachRunTime=(len(WL_file[:,0])-1)*3600.
XBX,XBY,XBZ,NE = xb.xbeach_grid_bed_setup(ProfileFile,Dsf,Slope,StructureElevation,ErosionEnabled,CliffedBeach)
xb.WriteParams(XBeachRunTime,(len(XBX)-1),XBX[0],StructureElevation,ErosionEnabled)

fig = plt.figure(1)
ax1=plt.subplot(111)

ax1.plot(XBX,XBZ,'-xk',markersize=2.5)
ax1.plot([XBX[0],XBX[-1]],[MaxWL,MaxWL],'b',label='Max. water level')
ax1.plot([XBX[0],XBX[-1]],[MinWL,MinWL],'b--',label='Min. water level')
ax1.set_xlabel('Cross-shore distance (m)')
ax1.set_ylabel('Elevation (m)')
ax1.legend(loc=4)

ax1.set_xlim(1200,2000)
ax1.set_ylim(-4,5)

plt.tight_layout()
pltname = wkspc + "Profile_minWL_maxWL.png"
plt.savefig(pltname, dpi = 300)
plt.close()

fig = plt.figure(2,figsize=(12,16))

ax1 = plt.subplot(311)

q_95 = numpy.quantile(tide_output[:,1],.975)

date_list = date_list[w_time]
ax1.plot(date_list,tide_output[w_time,1],linewidth=1.,color="dodgerblue",label="Atlantic City WL")
label1 = "97.5%ile WL\n" + str(numpy.round(q_95,1)) + " ft rel. MLLW"
ax1.plot([date_list[0],date_list[-1]],[q_95,q_95],linewidth=1.,color="red",label=label1)

ax1.plot(date_list[w_seq],tide_output[w_time[w_seq], 1],linewidth=1.5,color="tomato",label="WL(t)")

ax1.set_xlim(date_list[w_seq][0],date_list[w_seq][-1])
ax1.grid()
ax1.legend(loc=3)
ax1.set_ylabel("Water level (m. rel to MLLW)")

ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax1 = plt.subplot(312)

q_95 = numpy.quantile(wave_output[:,1],.975)

ax1.plot(date_list[w_seq],wave_output[w_time[w_seq],1],linewidth=1.,color="k",label="SWH(t)")

label1 = "97.5%ile SWH\n" + str(numpy.round(q_95,1)) + " m"
ax1.plot([date_list[0],date_list[-1]],[q_95,q_95],linewidth=1.,color="red",label=label1)

ax1.set_xlim(date_list[w_seq][0],date_list[w_seq][-1])
ax1.grid()
ax1.legend(loc=1)
ax1.set_ylabel("NOAA buoy 440915 waves (m.)")

ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax1 = plt.subplot(313)

ax1.plot(beach_profile[:,0],beach_profile[:,2],linewidth=1.,color="dodgerblue",label="SH8")

ax1.set_xlim(numpy.min(beach_profile[:,0]),numpy.max(beach_profile[:,0]))
ax1.grid()
ax1.legend(loc=1)
ax1.set_ylabel("Elevation (m)")
ax1.set_xlabel("Distance (m)")

plt.tight_layout()
pltname = wkspc + "Wave_Tide_data.png"
plt.savefig(pltname, dpi = 300)

plt.close()