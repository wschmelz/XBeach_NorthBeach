import os
import sys
import glob
import numpy
import pandas
import datetime
import time
import loess2D
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import scipy
from scipy import interpolate

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import random
import pwlf
import requests
import tarfile
import pandas as pd
from io import BytesIO
from dateutil import rrule

"""
Set workspace paths
"""
backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))
print (time.strftime("%H:%M:%S"))
print ()

"""
Define helper functions for RMSE and R-value calculations
"""

def calculate_rmse(y_true, y_pred):
    return numpy.sqrt(mean_squared_error(y_true, y_pred))
	
def calculate_r_value(y_true, y_pred):
    correlation_matrix = numpy.corrcoef(y_true, y_pred)
    r_value = correlation_matrix[0, 1]
    return r_value
	
"""
Define infill functions for missing data using various regression models
"""

def infill_missing_data_gbr(x_1, x_2, y_obs, x_1_missing, x_2_missing):
	X_train = numpy.column_stack((x_1, x_2))
	y_train = y_obs
	gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
	gbr_model.fit(X_train, y_train)
	X_missing = numpy.column_stack((x_1_missing, x_2_missing))
	y_predicted = gbr_model.predict(X_missing)
	y_predicted2 = gbr_model.predict(X_train)
	return y_predicted, y_predicted2

def infill_missing_data_rf(x_1, x_2, y_obs, x_1_missing, x_2_missing):
    X_train = numpy.column_stack((x_1, x_2))
    y_train = y_obs
    rf_model = RandomForestRegressor(n_estimators=500, random_state=0)
    rf_model.fit(X_train, y_train)
    X_missing = numpy.column_stack((x_1_missing, x_2_missing))
    y_predicted = rf_model.predict(X_missing)
    y_predicted2 = rf_model.predict(X_train)
    return y_predicted,y_predicted2


def infill_missing_data_nn(x_1, x_2, y_obs, x_1_missing, x_2_missing):
	X_train = numpy.column_stack((x_1, x_2))
	y_train = y_obs
	nn_model = MLPRegressor(hidden_layer_sizes=(100, 10), max_iter=10000, random_state=42)
	nn_model.fit(X_train, y_train)
	X_missing = numpy.column_stack((x_1_missing, x_2_missing))
	y_predicted = nn_model.predict(X_missing)
	y_predicted2 = nn_model.predict(X_train)
	return y_predicted, y_predicted2

def infill_missing_data_lr(x_1, x_2, y_obs, x_1_missing, x_2_missing):
	X_train = numpy.column_stack((x_1, x_2))
	y_train = y_obs
	lr_model = LinearRegression()
	lr_model.fit(X_train, y_train)
	X_missing = numpy.column_stack((x_1_missing, x_2_missing))
	y_predicted = lr_model.predict(X_missing)
	y_predicted2 = lr_model.predict(X_train)
	return y_predicted, y_predicted2

def infill_with_piecewise_linear(x_1, x_2, y_obs, x_1_missing, x_2_missing, n_segments=6):

    sorted_idx = numpy.argsort(x_2)
    x_sorted = x_2[sorted_idx]
    y_sorted = y_obs[sorted_idx]

    my_pwlf = pwlf.PiecewiseLinFit(x_sorted, y_sorted)
    res = my_pwlf.fit(n_segments)

    y_missing_pred = my_pwlf.predict(x_2_missing)
    y_train_pred = my_pwlf.predict(x_2)
    return y_missing_pred, y_train_pred

"""
Define a function that combines linear regression and random forest for infilling missing data
"""

def infill_missing_data(x_1, x_2, y_obs, x_1_missing, x_2_missing):
    y_predicted_lr, y_predicted_train_lr = infill_missing_data_lr(x_1, x_2, y_obs, x_1_missing, x_2_missing)
    residuals = y_obs - y_predicted_train_lr
    X_train = numpy.column_stack((x_1, x_2))
    rf_model = RandomForestRegressor(n_estimators=250, random_state=0)
    rf_model.fit(X_train, residuals)
    X_missing = numpy.column_stack((x_1_missing, x_2_missing))
    residuals_predicted = rf_model.predict(X_missing)
    residuals_predicted_train = rf_model.predict(X_train)
    y_final_predicted2 = y_predicted_train_lr + residuals_predicted_train
    y_final_predicted = y_predicted_lr + residuals_predicted
    rmse_lr = calculate_rmse(y_obs, y_predicted_train_lr)
    r_value_lr = calculate_r_value(y_obs, y_predicted_train_lr)
    rmse_combined = calculate_rmse(y_obs, y_final_predicted2)
    r_value_combined = calculate_r_value(y_obs, y_final_predicted2)
    print("Linear Regression RMSE:", rmse_lr)
    print("Linear Regression R value:", r_value_lr)
    print("Combined RMSE:", rmse_combined)
    print("Combined R value:", r_value_combined)
    return y_final_predicted, y_final_predicted2

"""
Hindcast data download functions
"""

def fetch_multi_1_data(
    station_id='44025',
    start_date='2010-01',
    end_date='2019-12'
):

    if len(start_date) == 7:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m")
    else:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    if len(end_date) == 7:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m")
    else:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    pivot_dt = datetime.datetime(2015, 8, 1)

    all_rows = []

    for dt_month in rrule.rrule(rrule.MONTHLY, dtstart=start_dt, until=end_dt):
        yyyymm = dt_month.strftime('%Y%m')
        year_int = dt_month.year
        month_int = dt_month.month

        base_url = (
            "https://polar.ncep.noaa.gov/waves/hindcasts/multi_1/"
            f"{yyyymm}/points"
        )
        tar_fname = f"multi_1_base.buoys_wmo.{yyyymm}.tar.gz"
        tar_url = f"{base_url}/{tar_fname}"

        print(f"[INFO] Downloading {tar_url}")

        if dt_month < pivot_dt:
            target_filename = f"multi_1.{station_id}.HIND_WMO.{yyyymm}"
        else:
            target_filename = f"multi_1.{station_id}.WMO.{yyyymm}"

        try:
            resp = requests.get(tar_url, timeout=60)
            if resp.status_code != 200:
                print(f"  -> Not found (HTTP {resp.status_code}). Skipping {yyyymm}...")
                continue

            tar_bytes = BytesIO(resp.content)
            with tarfile.open(fileobj=tar_bytes, mode='r:*') as outer_tar:
                members = outer_tar.getmembers()
                if not members:
                    print("  -> Outer TAR is empty. Skipping...")
                    continue

                found_member = None
                for m in members:
                    if m.isfile() and m.name.endswith(target_filename):
                        found_member = m
                        break

                if found_member:
                    fobj = outer_tar.extractfile(found_member)
                    if not fobj:
                        print(f"  -> Could not extract {found_member.name}")
                        continue

                    lines = fobj.read().decode('utf-8', errors='ignore').splitlines()
                    parse_multi_1_lines(lines, all_rows)

                else:
                    inner_tar_member = None
                    for m in members:
                        if m.isfile() and m.name.endswith('.tar'):
                            inner_tar_member = m
                            break

                    if not inner_tar_member:
                        print(f"  -> Station file {target_filename} not found in outer or inner tar.")
                        continue

                    inner_tar_data = outer_tar.extractfile(inner_tar_member).read()
                    with tarfile.open(fileobj=BytesIO(inner_tar_data), mode='r:*') as inner_tar:
                        inner_member = None
                        for m2 in inner_tar.getmembers():
                            if m2.isfile() and m2.name.endswith(target_filename):
                                inner_member = m2
                                break

                        if not inner_member:
                            print(f"  -> Station file {target_filename} not found in inner tar.")
                            continue

                        fobj2 = inner_tar.extractfile(inner_member)
                        if not fobj2:
                            print(f"  -> Could not extract {inner_member.name}")
                            continue

                        lines = fobj2.read().decode('utf-8', errors='ignore').splitlines()
                        parse_multi_1_lines(lines, all_rows)

        except Exception as ex:
            print(f"  -> ERROR fetching {tar_url}: {ex}")
            continue

    if not all_rows:
        print("[INFO] No data found for the specified date range/station.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.sort_values('DateTimeUTC', inplace=True)
    df.set_index('DateTimeUTC', inplace=True)

    return df


def parse_multi_1_lines(lines, all_rows):

    for ln in lines:
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            yy = int(parts[0])
            mm = int(parts[1])
            dd = int(parts[2])
            hh = int(parts[3])
            mwd = float(parts[5])
            swh = float(parts[6])
            period = float(parts[7])

            dt_line = datetime.datetime(yy, mm, dd, hh)
            all_rows.append({
                'DateTimeUTC': dt_line,
                'SWH': swh,
                'MWD': mwd,
                'Period': period
            })
        except ValueError:
            continue

def fetch_nopp_phase2_data(
    station_id='44025',
    start_date='1979-01',
    end_date='2009-12'
):
    """
    Mirrors fetch_multi_1_data() except it pulls from the nopp-phase2 archive.
    Only differences: base URL and tar file naming.
    """
    # Parse start_date / end_date similarly
    if len(start_date) == 7:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m")
    else:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    if len(end_date) == 7:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m")
    else:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    all_rows = []

    # Loop over each month, just like fetch_multi_1_data
    for dt_month in rrule.rrule(rrule.MONTHLY, dtstart=start_dt, until=end_dt):
        yyyymm = dt_month.strftime('%Y%m')
        year_int = dt_month.year
        month_int = dt_month.month

        # Base URL for nopp-phase2
        base_url = (
            "https://polar.ncep.noaa.gov/waves/hindcasts/nopp-phase2/"
            f"{yyyymm}/points/buoys"
        )

        # NOPP Phase 2 tar file naming
        tar_fname = f"multi_reanal.buoys_wmo.buoys.{yyyymm}.tar.gz"
        tar_url = f"{base_url}/{tar_fname}"

        print(f"[INFO] Downloading {tar_url}")

        target_filename = f"multi_reanal.{station_id}.WMO.{yyyymm}"
        
        try:
            resp = requests.get(tar_url, timeout=60)
            if resp.status_code != 200:
                print(f"  -> Not found (HTTP {resp.status_code}). Skipping {yyyymm}...")
                continue

            tar_bytes = BytesIO(resp.content)
            with tarfile.open(fileobj=tar_bytes, mode='r:*') as outer_tar:
                members = outer_tar.getmembers()
                if not members:
                    print("  -> Outer TAR is empty. Skipping...")
                    continue

                found_member = None
                for m in members:
                    if m.isfile() and m.name.endswith(target_filename):
                        found_member = m
                        break

                if found_member:
                    fobj = outer_tar.extractfile(found_member)
                    if not fobj:
                        print(f"  -> Could not extract {found_member.name}")
                        continue

                    lines = fobj.read().decode('utf-8', errors='ignore').splitlines()
                    parse_multi_1_lines(lines, all_rows)

                else:
                    inner_tar_member = None
                    for m in members:
                        if m.isfile() and m.name.endswith('.tar'):
                            inner_tar_member = m
                            break

                    if not inner_tar_member:
                        print(f"  -> Station file {target_filename} not found in outer or inner tar.")
                        continue

                    inner_tar_data = outer_tar.extractfile(inner_tar_member).read()
                    with tarfile.open(fileobj=BytesIO(inner_tar_data), mode='r:*') as inner_tar:
                        inner_member = None
                        for m2 in inner_tar.getmembers():
                            if m2.isfile() and m2.name.endswith(target_filename):
                                inner_member = m2
                                break

                        if not inner_member:
                            print(f"  -> Station file {target_filename} not found in inner tar.")
                            continue

                        fobj2 = inner_tar.extractfile(inner_member)
                        if not fobj2:
                            print(f"  -> Could not extract {inner_member.name}")
                            continue

                        lines = fobj2.read().decode('utf-8', errors='ignore').splitlines()
                        parse_multi_1_lines(lines, all_rows)

        except Exception as ex:
            print(f"  -> ERROR fetching {tar_url}: {ex}")
            continue

    if not all_rows:
        print("[INFO] No data found for the specified date range/station.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.sort_values('DateTimeUTC', inplace=True)
    df.set_index('DateTimeUTC', inplace=True)

    return df

def convert_df_to_wave_data(df):

    n = len(df)
    wave_data = numpy.full((n, 12), numpy.nan)

    for i, (timestamp, row) in enumerate(df.iterrows()):
        wave_data[i, 0] = timestamp.year
        wave_data[i, 1] = timestamp.month
        wave_data[i, 2] = timestamp.day
        wave_data[i, 3] = timestamp.hour
        wave_data[i, 4] = 0.0

        wave_data[i, 8] = row['SWH']     # SWH
        wave_data[i, 9] = row['Period']  # wave period
        wave_data[i,11] = row['MWD']     # wave direction

    return wave_data

"""
Download tide and wave data using requests, and store them locally
"""

buoy_num = 44065
station_num = 8531680
MTK_buoy_num = 44017 
battery_station_ID = 8518750
AC_station_ID = 8534720

tide_dir = wkspc + f"01_Station{station_num}/"
wave_dir = wkspc + f"00_Buoy{buoy_num}/"
base_url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date={year}0101&end_date={year}1231&datum=MLLW&station={station_num}&time_zone=LST&units=english&format=csv")
os.makedirs(tide_dir, exist_ok=True)
year_start = 1979
year_end = 2023

for year in range(year_start, year_end):
    full_url = base_url.format(year=year,station_num=station_num)
    response = requests.get(full_url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(tide_dir, f"tide_data_{year}.csv"), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded tide file for year {year}")
    else:
        print(f"Failed to download tide file for year {year}")

print("Tide download completed.")

base_url = "https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_num}h{year}.txt.gz&dir=data/historical/stdmet/"
os.makedirs(wave_dir, exist_ok=True)

for year in range(year_start, year_end):
    full_url = base_url.format(year=year,buoy_num=buoy_num)
    response = requests.get(full_url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(wave_dir, f"{buoy_num}h{year}.txt"), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded wave file for year {year}")
    else:
        print(f"Failed to download wave file for year {year}")

print("Wave download completed.")

buoy_num = int(MTK_buoy_num * 1.0)
station_num = int(battery_station_ID * 1.0)
tide_dir = wkspc + f"01_Station{station_num}/"
wave_dir = wkspc + f"00_Buoy{buoy_num}/"
base_url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date={year}0101&end_date={year}1231&datum=MLLW&station={station_num}&time_zone=LST&units=english&format=csv")
os.makedirs(tide_dir, exist_ok=True)


for year in range(year_start, year_end):
    full_url = base_url.format(year=year,station_num=station_num)
    response = requests.get(full_url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(tide_dir, f"tide_data_{year}.csv"), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded tide file for year {year}")
    else:
        print(f"Failed to download tide file for year {year}")

print("Tide download completed.")

base_url = "https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_num}h{year}.txt.gz&dir=data/historical/stdmet/"
os.makedirs(wave_dir, exist_ok=True)

for year in range(year_start, year_end):
    full_url = base_url.format(year=year,buoy_num=buoy_num)
    response = requests.get(full_url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(wave_dir, f"{buoy_num}h{year}.txt"), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded wave file for year {year}")
    else:
        print(f"Failed to download wave file for year {year}")

print("Wave download completed.")

"""
Define functions to load tide and wave data from disk
"""

def load_tide_data(workspace):
	tide_datafiles = glob.glob(workspace + "/*.csv")
	count = 0 
	for file_out in tide_datafiles:
		df = pandas.read_csv(file_out,skiprows=1, header=None)
		tide_data_tmp1 = numpy.array(df)
		x1 = pandas.to_datetime(tide_data_tmp1[:,0])
		tide_data_tmp = numpy.zeros((len(tide_data_tmp1[:,0]),5))
		tide_data_tmp[:,0] = x1.year
		tide_data_tmp[:,1] = x1.month
		tide_data_tmp[:,2] = x1.day
		tide_data_tmp[:,3] = x1.hour
		tide_data_tmp[:,4] = tide_data_tmp1[:,1]
		if count ==0:
			tide_data = tide_data_tmp
		else:
			tide_data = numpy.append(tide_data,tide_data_tmp,axis=0)
		count+=1
	return tide_data
	
def load_wave_data(workspace):
	wave_datafiles = glob.glob(workspace + "/*.txt")
	count = 0   
	for file_out in wave_datafiles:
		wave_data_tmp = numpy.genfromtxt(file_out,skip_header=2)
		if count == 0:
			wave_data = wave_data_tmp * 1.0
		else:
			wave_data = numpy.append(wave_data,wave_data_tmp,axis=0)	
		count+=1
	return wave_data
	
"""
Define functions to process downloaded tide and wave data
"""
def process_tide(tide_data):
	t1 = time.time()
	tide_output_tmp = numpy.zeros((len(tide_data[:,0]),2))
	date_list = []
	for n2 in range(0,len(tide_data[:,0])):	
		year_tmp = int(tide_data[n2,0])
		month_tmp = int(tide_data[n2,1])
		day_tmp = int(tide_data[n2,2]) 
		hour_tmp = int(tide_data[n2,3])
		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
		time_tmp_years = (time_datum_stop - time_datum_start).total_seconds() / (86400*365.25)
		date_list.append([time_datum_stop,])
		tide_output_tmp[n2,0] = time_tmp
		tide_output_tmp[n2,1] = tide_data[n2,4]
	w1 = numpy.where((tide_output_tmp[:,0]>=time_tmp_first)&(tide_output_tmp[:,0]<time_tmp_last))[0]
	tide_output = tide_output_tmp[w1,:]
	date_list = numpy.array(date_list)[w1]
	w2 = numpy.where(numpy.isnan(tide_output[:,1])==False)[0]
	t2 = time.time()
	print("Minutes to process tide data:",(t2-t1)/60.)
	print("All n, tide data:",len(tide_output[:,1]))
	print("Valid n, tide data:",len(w2))
	print("Percent n valid:",round((float(len(w2))/len(tide_output[:,1]))*100.,2))
	print()
	return tide_output, tide_output_tmp, date_list	
	
def process_wave(wave_data,tide_output_tmp):
	t1 = time.time()
	wave_data_tmp = wave_data.copy()
	w1 = numpy.where((wave_data_tmp[:,8]<90.)&(wave_data_tmp[:,11]<400.))[0]
	wave_data = wave_data_tmp[w1,:] * 1.0
	wave_output_tmp = numpy.zeros((len(wave_data[:,0]),5))
	w1 = numpy.where((wave_data[:,11]<90.)&(wave_data[:,11]>=0.))[0]
	w2 = numpy.where((wave_data[:,11]<180.)&(wave_data[:,11]>=90.))[0]
	w3 = numpy.where((wave_data[:,11]<270.)&(wave_data[:,11]>=180.))[0]
	w4 = numpy.where((wave_data[:,11]<360.)&(wave_data[:,11]>=270.))[0]
	wave_output_tmp[w1,1] = (wave_data[w1,11] * -1.) + 90.
	wave_output_tmp[w2,1] = (wave_data[w2,11] * -1.) + 450.
	wave_output_tmp[w3,1] = (wave_data[w3,11] * -1.) + 450.
	wave_output_tmp[w4,1] = (wave_data[w4,11] * -1.) + 450.
	baseline_ang = 90.
	wave_output_ang_tmp = wave_output_tmp[:,1] + (90. - baseline_ang)
	w1 = numpy.where(wave_output_ang_tmp>360)[0]
	wave_output_ang_tmp[w1] = wave_output_ang_tmp[w1] - 360
	'''
	w2 = numpy.where((wave_output_ang_tmp<270.)&(wave_output_ang_tmp>90.))[0]
	wave_data[w2,8] = wave_data[w2,8]*0.0
	'''
	wave_output_tmp_2 = numpy.zeros((len(tide_output_tmp[:,0]),5)) * numpy.nan
	for n2 in range(0,len(wave_data[:,0])):	
		year_tmp = int(wave_data[n2,0])
		month_tmp = int(wave_data[n2,1])
		day_tmp = int(wave_data[n2,2])
		hour_tmp = int(wave_data[n2,3])
		min_tmp = int(wave_data[n2,4])
		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
		w_tidematch = numpy.where(numpy.absolute(tide_output_tmp[:,0]-time_tmp)<0.25/24.)[0]
		if len(w_tidematch) > 0:
			wave_output_tmp_2[w_tidematch,0] = time_tmp
			wave_output_tmp_2[w_tidematch,1] = wave_output_tmp[n2,1]
			wave_output_tmp_2[w_tidematch,2] = wave_data[n2,8]
			wave_output_tmp_2[w_tidematch,3] = wave_data[n2,9]
	w2 = numpy.where(numpy.isnan(wave_output_tmp_2[:,1])==False)[0]
	w_tmp2 = numpy.where(numpy.isnan(wave_output_tmp_2[:,2])==False)[0]
	w_tmp3 = numpy.where(numpy.isnan(wave_output_tmp_2[:,3])==False)[0]
	wave_output_tmp_2[:,0] = tide_output_tmp[:,0] * 1.0
	time_wave = wave_output_tmp_2[:,0]*1.0
	wave_output_tmp = numpy.zeros((len(wave_output_tmp_2[:,0]),4)) * numpy.nan
	wave_output_tmp_linint = numpy.zeros((len(wave_output_tmp_2[:,0]),4)) * numpy.nan
	w_per_0 = numpy.where(wave_output_tmp_2[:,3]==0.)[0]
	w_per_non0 = numpy.where(wave_output_tmp_2[:,3]>0.)[0]
	wave_output_tmp_2[w_per_0,3] = numpy.nanmean(wave_output_tmp_2[w_per_non0,3])
	f_swh = scipy.interpolate.interp1d(wave_output_tmp_2[w_tmp2,0],wave_output_tmp_2[w_tmp2,2])
	f_period = scipy.interpolate.interp1d(wave_output_tmp_2[w_tmp3,0],wave_output_tmp_2[w_tmp3,3])
	
	w_tmp4 = numpy.where(time_wave<=numpy.max(wave_output_tmp_2[w_tmp2,0]))[0]
	
	wave_output_tmp[:,0] = time_wave *1.0
	
	wave_output_tmp_linint[w_tmp4,1] = f_swh(time_wave[w_tmp4])
	wave_output_tmp[w_tmp4,1] = loess2D.weighted_linear(time_wave[w_tmp4],time_wave[w_tmp4]*0.0,wave_output_tmp_2[w_tmp2,0],wave_output_tmp_2[w_tmp2,0]*0.0,wave_output_tmp_2[w_tmp2,2],3,3.0/24.,3.0/24.)[2]
	wave_output_tmp_linint[w_tmp4,2] = f_period(time_wave[w_tmp4])
	wave_output_tmp[w_tmp4,2] = loess2D.weighted_linear(time_wave[w_tmp4],time_wave[w_tmp4]*0.0,wave_output_tmp_2[w_tmp3,0],wave_output_tmp_2[w_tmp3,0]*0.0,wave_output_tmp_2[w_tmp3,3],3,3.0/24.,3.0/24.)[2]
	
	w1 = numpy.where((wave_output_tmp[:,0]>=time_tmp_first)&(wave_output_tmp[:,0]<time_tmp_last))[0]
	w_nan = numpy.where(numpy.isnan(wave_output_tmp[:,1]))[0]
	w2 = numpy.where(numpy.isnan(wave_output_tmp[:,2]))[0]
	w3 = numpy.where(numpy.isnan(wave_output_tmp[:,3]))[0]
	wave_output_tmp[w_nan,3] = wave_output_tmp[w_nan,3] + 1.
	wave_output = wave_output_tmp[w1,:]* 1.0
	wave_output_linint = wave_output_tmp_linint[w1,:]* 1.0
	date_list = []
	for n2 in range(0,len(wave_output_tmp[:,0])):	
		time_datum_stop = time_datum_start + datetime.timedelta(days = wave_output_tmp[n2,0])
		date_list.append([time_datum_stop,])
	date_list = numpy.array(date_list)[w1]
	w1 = numpy.where((wave_output[:,0]>=time_tmp_first)&(wave_output[:,0]<time_tmp_last))[0]
	wave_output = wave_output[w1,:]* 1.0
	wave_output_linint = wave_output_linint[w1,:]* 1.0
	date_list = numpy.array(date_list)[w1]
	w2 = numpy.where(numpy.isnan(wave_output[:,1])==False)[0]
	t2 = time.time()
	print("Minutes to process wave data:",(t2-t1)/60.)
	print("All n, wave data:",len(wave_output[:,1]))
	print("Valid n, wave data:",len(w2))
	print("Percent n valid:",round((float(len(w2))/len(wave_output[:,1]))*100.,2))
	print()
	return wave_output, date_list, wave_output_linint

"""
Define relevant time period for processing
"""
year_tmp = 2009
month_tmp = 9
day_tmp = 15
hour_tmp = 0
time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp,hour_tmp)

year_tmp2 = 2023
month_tmp = 1
day_tmp = 1
hour_tmp = 0
time_datum_last_data = datetime.datetime(year_tmp2, month_tmp, day_tmp,hour_tmp)

year_tmp_1 = 1979
month_tmp = 1
day_tmp = 1
hour_tmp = 0
time_datum_first_data = datetime.datetime(year_tmp_1, month_tmp, day_tmp,hour_tmp)

time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)

cutoff2_date = datetime.datetime(2019, 6, 1, 0, 0)
time_tmp_cutoff2 = (cutoff2_date - time_datum_start).total_seconds() / (3600.0*24.)

cutoff_nopp2_date = datetime.datetime(2008, 1, 1, 0, 0)
time_tmp_cutoff_nopp2 = (cutoff_nopp2_date - time_datum_start).total_seconds() / (3600.0 * 24.0)

buoy_num = 44065
station_num = 8531680
tide_dir = wkspc + f"01_Station{station_num}/"
wave_dir = wkspc + f"00_Buoy{buoy_num}/"

"""
Process tide data
"""

tide_datafiles = glob.glob(tide_dir + "*.csv")
tide_data = load_tide_data(tide_dir)
filename = wkspc+ f"Station{station_num}_Tides_{year_tmp_1}_{year_tmp2}.txt"
numpy.savetxt(filename, tide_data, delimiter=',')
tide_output, tide_output_tmp, date_list_tide = process_tide(tide_data)
tide_file = wkspc + 'tide_output_orig.csv'
numpy.savetxt(tide_file, tide_output, delimiter=',')

tide_file = wkspc + 'tide_output_orig.csv'
tide_output = numpy.genfromtxt(tide_file,delimiter=',')

"""
Process wave data
"""

df_nopp2 = fetch_nopp_phase2_data(
    station_id='44025',
    start_date='1979-01',
    end_date='2009-12'
)

print(df_nopp2.head())
print(df_nopp2.tail())
print("Total rows:", len(df_nopp2))

# 2) Convert to the same “wave_data” array format as Multi_1

df_data = fetch_multi_1_data(
	station_id='44025',
	start_date='2008-01',
	end_date='2022-12'
)
print(df_data.head())
print(df_data.tail())
print("Total rows:", len(df_data))

wave_data_nopp2 = convert_df_to_wave_data(df_nopp2)
wave_data_mulit1 = convert_df_to_wave_data(df_data)


wave_output_tmp_nopp2 = numpy.zeros((len(wave_data_nopp2[:,0]), 1))

for n in range(len(wave_data_nopp2[:,0])):
    year_tmp  = int(wave_data_nopp2[n,0])
    month_tmp = int(wave_data_nopp2[n,1])
    day_tmp   = int(wave_data_nopp2[n,2])
    hour_tmp  = int(wave_data_nopp2[n,3])
    min_tmp   = int(wave_data_nopp2[n,4])

    time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
    time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0 * 24.0)

    # Keep row if it’s strictly before 2009-01-01
    if time_tmp < time_tmp_cutoff_nopp2:
        # Also check the same tide matching condition you used for multi_1:
        w_tidematch = numpy.where(
            (numpy.absolute(tide_output_tmp[:,0] - time_tmp) <= 0.5/24.)
        )[0]
        if len(w_tidematch) > 0:
            wave_output_tmp_nopp2[n, 0] = 1

# (D) Filter to “keep” rows indicated by 1
w_nopp2_indices = numpy.where(wave_output_tmp_nopp2 == 1)[0]
wave_data_nopp2 = wave_data_nopp2[w_nopp2_indices, :]*1.

print("filtered wave_data_nopp2 shape:", wave_data_nopp2.shape)

wave_output_tmp_multi1 = numpy.zeros((len(wave_data_mulit1[:,0]),1))
for n in range(0,len(wave_data_mulit1[:,0])):	
	year_tmp = int(wave_data_mulit1[n,0])
	month_tmp = int(wave_data_mulit1[n,1])
	day_tmp = int(wave_data_mulit1[n,2])
	hour_tmp = int(wave_data_mulit1[n,3])
	min_tmp = int(wave_data_mulit1[n,4])
	time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
	time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
	w_tidematch = numpy.where((numpy.absolute(tide_output_tmp[:,0]-time_tmp)<=0.5/24.) & (time_tmp<time_tmp_cutoff2))[0]
	if len(w_tidematch) > 0:
		wave_output_tmp_multi1[n,0] = 1
w_wave_data_multi1 = numpy.where(wave_output_tmp_multi1==1)	[0]
wave_data_mulit1 = wave_data_mulit1[w_wave_data_multi1,:]*1.

print("filtered wave_data_mulit1 shape:", wave_data_mulit1.shape)

wave_data_nbdc = load_wave_data(wave_dir)

wave_output_tmp_nbdc = numpy.zeros((len(wave_data_nbdc[:,0]),1))
for n in range(0,len(wave_data_nbdc[:,0])):	
	year_tmp = int(wave_data_nbdc[n,0])
	month_tmp = int(wave_data_nbdc[n,1])
	day_tmp = int(wave_data_nbdc[n,2])
	hour_tmp = int(wave_data_nbdc[n,3])
	min_tmp = int(wave_data_nbdc[n,4])
	time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)
	time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
	w_tidematch = numpy.where((numpy.absolute(tide_output_tmp[:,0]-time_tmp)<=0.5/24.) & (time_tmp>=time_tmp_cutoff2))[0]
	if len(w_tidematch) > 0:
		wave_output_tmp_nbdc[n,0] = 1
		
w_wave_data_nbdc = numpy.where(wave_output_tmp_nbdc==1)	[0]
wave_data_nbdc = wave_data_nbdc[w_wave_data_nbdc,0:12]*1.

print("filtered wave_data_nbdc shape:", wave_data_nbdc.shape)

filename = wkspc +f"Multi1_Buoy{buoy_num}_Waves_{year_tmp_1}_{year_tmp2}.txt"
numpy.savetxt(filename, wave_data_mulit1)
filename = wkspc +f"NBDC_Buoy{buoy_num}_Waves_{year_tmp_1}_{year_tmp2}.txt"
numpy.savetxt(filename, wave_data_nbdc)

wave_data = numpy.append(wave_data_nopp2,wave_data_mulit1,axis=0)
print("wave_data shape:", wave_data.shape)
wave_data = numpy.append(wave_data,wave_data_nbdc,axis=0)

print("wave_data shape:", wave_data.shape)

wave_output, date_list_wave, wave_output_linint = process_wave(wave_data,tide_output_tmp)

wave_file = wkspc + 'wave_data.csv'
numpy.savetxt(wave_file,wave_output, delimiter=',')

wave_file = wkspc + 'wave_data.csv'
wave_output = numpy.genfromtxt(wave_file,delimiter=',')

battery_station_ID = 8518750
station_num = int(battery_station_ID * 1.0)
tide_dir_TB = wkspc + f"01_Station{station_num}/"

"""
Process second set of tide data
"""

tide_datafiles = glob.glob(tide_dir_TB + "*.csv")
tide_data_TB = load_tide_data(tide_dir_TB)
filename = wkspc+ f"Station{station_num}_Tides_{year_tmp_1}_{year_tmp2}.txt"
numpy.savetxt(filename, tide_data_TB, delimiter=',')
tide_output_TB, tide_output_tmp_TB, date_list_tide_TB = process_tide(tide_data_TB)
tide_file = wkspc + 'tide_output_TB_orig.csv'
numpy.savetxt(tide_file, tide_output_TB, delimiter=',')

tide_file = wkspc + 'tide_output_TB_orig.csv'
tide_output_TB = numpy.genfromtxt(tide_file,delimiter=',')

"""
Perform infill for tide data
"""

w_notnan_SH_TB = numpy.where((numpy.isnan(tide_output[:,1])==False)&(numpy.isnan(tide_output_TB[:,1])==False))[0]
w_nan_SH = numpy.where((numpy.isnan(tide_output[:,1])))[0]
w_nan_SH2 = numpy.where((numpy.isnan(tide_output[:,1])==False))[0]
x_1 = tide_output[w_notnan_SH_TB, 0]
x_2 = tide_output_TB[w_notnan_SH_TB, 1]
y_obs = tide_output[w_notnan_SH_TB, 1]
perc_valid = (len(w_nan_SH)/len(tide_output[:,1]))*100.
print ("tide percentage invalid:",str(round(perc_valid,2)))
x_1_missing = tide_output[w_nan_SH, 0]
x_2_missing = tide_output_TB[w_nan_SH, 1]
y_new,y_new_2 = infill_missing_data(x_1, x_2, y_obs, x_1_missing, x_2_missing)
tide_output[w_nan_SH,1] = y_new
fig = plt.figure(1,figsize=(12,9))
ax1 = plt.subplot(311)
tide_output_pltcopy = tide_output.copy()
tide_output_pltcopy[w_notnan_SH_TB,1] = y_new_2
ax1.plot(date_list_tide,tide_output[:,1], linewidth = 1.0,color='k')
ax1.plot(date_list_tide,tide_output_pltcopy[:,1], linewidth = 0.5,color='dodgerblue')
ax1.plot(date_list_tide[w_nan_SH],tide_output[w_nan_SH,1], linewidth = 0.5,color='gray')
residuals = y_obs - y_new_2
residuals_stdev = calculate_rmse(y_obs,y_new_2)

print()
print("tide rmse",residuals_stdev)

w2 = numpy.where(numpy.isnan(tide_output[:,1])==False)[0]

print("All n, tide data:",len(tide_output[:,1]))
print("Valid n, tide data:",len(w2))
print("Percent n valid, tide data:",round((float(len(w2))/len(tide_output[:,1]))*100.,2))
print()

w2 = numpy.where(numpy.isnan(wave_output[:,1])==False)[0]
perc_valid = (len(w2)/len(wave_output[:,1]))*100.

print("All n, wave data:",len(wave_output[:,1]))
print("Valid n, wave data:",len(w2))
print("Percent n valid, wave data:",round((float(len(w2))/len(wave_output[:,1]))*100.,2))
print()

ax1.set_ylabel("wl (ft re MLLW)")
ax1.set_xticklabels([])
ax1.set_xlim(date_list_tide[0],date_list_tide[-1])
ax1 = plt.subplot(312)

ax1.plot(date_list_tide,wave_output_linint[:,1], linewidth = 0.5,color='gray')
ax1.plot(date_list_tide,wave_output[:,1], linewidth = 0.5,color='tomato')

w2 = numpy.where((numpy.isnan(wave_output[:,1]))|(numpy.isnan(wave_output[:,2])))[0]
wave_output[w2,3] = 1.0
w3 = numpy.where((numpy.isnan(wave_output[:,1])==False)&(numpy.isnan(wave_output[:,2])==False))[0]
wave_output[w3,3] = 0.0
ax1.set_xticklabels([])
ax1.set_xlim(date_list_tide[0],date_list_tide[-1])
ax1.set_ylim(0,10)
ax1.set_ylabel("swh (m)")
ax1 = plt.subplot(313)

ax1.plot(date_list_tide,wave_output[:,2], linewidth = 0.5,color='tomato')
ax1.set_ylabel("period (s)")
ax1.set_xlabel("Date")

"""
Save final processed tide and wave data if dates match
"""

tide_file = wkspc + 'tide_output_1979_2022.csv'
numpy.savetxt(tide_file, tide_output, delimiter=',')
wave_file = wkspc + 'wave_output_1979_2022.csv'
numpy.savetxt(wave_file,wave_output, delimiter=',')
date_list_tide_str = []
for idx in date_list_tide:
	for dt in idx:
		date_list_tide_str.append(dt.strftime('%Y-%m-%d %H:%M:%S'))	
date_list_file = wkspc + 'date_list.csv'
numpy.savetxt(date_list_file, date_list_tide_str, fmt='%s', delimiter=',')

ax1.set_ylim(0,25)
ax1.set_xlim(date_list_tide[0],date_list_tide[-1])
plt.tight_layout()
plt.savefig(f'Wave_Tide.png')
print("Data processed")
plt.show()
plt.close()
'''
"""
Load saved data files for interpolation and final plotting
"""
backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
tide_file = 'tide_output.csv'
wave_file = 'wave_output.csv'
tide_output = numpy.loadtxt(tide_file, delimiter=',')
wave_output = numpy.loadtxt(wave_file, delimiter=',')

def convert_seconds_to_real_time(seconds, start_time):
    return [start_time + datetime.timedelta(days=s) for s in seconds]

def interpolate_nans_w(data, max_gap=1.5):
	time = data[:, 0]
	values = data[:, 1]
	values2 = data[:, 2]
	nans = numpy.isnan(values)
	not_nans = ~nans
	for i in range(len(values) - 1):
		if nans[i]:
			start_idx = i - 1
			while i < len(values) and nans[i]:
				i += 1
			end_idx = i
			if end_idx < len(values) and (time[end_idx] - time[start_idx] <= max_gap):
				values[start_idx + 1:end_idx] = numpy.interp(
					time[start_idx + 1:end_idx],
					[time[start_idx], time[end_idx]],
					[values[start_idx], values[end_idx]]
				)
				values2[start_idx + 1:end_idx] = numpy.interp(
					time[start_idx + 1:end_idx],
					[time[start_idx], time[end_idx]],
					[values2[start_idx], values2[end_idx]]
				)
	return data

def interpolate_nans(data, max_gap=1.5):
	time = data[:, 0]
	values = data[:, 1]
	nans = numpy.isnan(values)
	not_nans = ~nans
	for i in range(len(values) - 1):
		if nans[i]:
			start_idx = i - 1
			while i < len(values) and nans[i]:
				i += 1
			end_idx = i
			if end_idx < len(values) and (time[end_idx] - time[start_idx] <= max_gap):
				values[start_idx + 1:end_idx] = numpy.interp(
					time[start_idx + 1:end_idx],
					[time[start_idx], time[end_idx]],
					[values[start_idx], values[end_idx]]
				)
	return data

start_time = datetime.datetime(2009, 9, 15, 0)
tide_time_real = convert_seconds_to_real_time(tide_output[:, 0], start_time)
wave_time_real = convert_seconds_to_real_time(wave_output[:, 0], start_time)
tide_output_infilled = interpolate_nans(tide_output.copy())
wave_output_infilled = interpolate_nans_w(wave_output.copy())
w1 =numpy.where(numpy.isnan(tide_output_infilled[:,1]))[0]
w2 =numpy.where((numpy.isnan(wave_output_infilled[:,1]))|(numpy.isnan(wave_output_infilled[:,2])))[0]
wave_output_infilled[:,3] = wave_output_infilled[:,3]*0.0
wave_output_infilled[w2,3] = (wave_output_infilled[w2,3]*0.0) + 1.
tide_file = wkspc + 'tide_output_interp.csv'
numpy.savetxt(tide_file, tide_output_infilled, delimiter=',')
wave_file = wkspc + 'wave_output_interp.csv'
numpy.savetxt(wave_file,wave_output_infilled, delimiter=',')

"""
Generate comparison plots of original vs infilled tide and wave data
"""
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
ax1.plot(tide_time_real, tide_output_infilled[:, 1], linewidth=0.5, linestyle='--', color='r', label='Infilled Tides')
ax1.plot(tide_time_real, tide_output[:, 1], linewidth=1.0, color='k', label='Original Tides')
ax1.set_title('Tides Over Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Tide Level')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.grid(True, which='both')

ax2.plot(wave_time_real, wave_output_infilled[:, 1], linewidth=0.5, linestyle='-', color='r', label='Infilled Waves')
ax2.plot(wave_time_real, wave_output[:, 1], linewidth=1.0, color='k', label='Original Waves')
ax2.set_title('Waves Over Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Wave Level')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.grid(True, which='both')

ax3.plot(wave_time_real, wave_output_infilled[:, 2], linewidth=0.5, linestyle='-', color='r', label='Infilled Waves')
ax3.plot(wave_time_real, wave_output[:, 2], linewidth=1.0, color='k', label='Original Waves')
ax3.set_title('Waves Over Time')
ax3.set_xlabel('Time')
ax3.set_ylabel('Wave Level')
ax3.legend()
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_minor_locator(mdates.MonthLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.grid(True, which='both')

plt.tight_layout()
plt.savefig('waves_and_tides_comparison.png')
plt.show()
plt.close()
'''