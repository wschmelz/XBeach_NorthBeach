import os
import datetime
import shutil
import time
import glob
import numpy
import sys
from scipy.integrate import simps

task_id = sys.argv[1]
backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
print(wkspc)
print(str(__file__))
start_time = datetime.datetime.now()
print("Time started:", time.strftime("%H:%M:%S"))

wkspc_DataPrep = wkspc + "00_DataPrep"
wkspc_Analysis = wkspc + "01_Analysis"
wkspc_Output = wkspc + "02_Output"
iters = 1

def loess(x_interp, t_series_x, t_series_z, pt_min_r, dist_r, factor):
    output = numpy.zeros(len(x_interp))
    for location in range(len(output)):
        new_z = "nan"
        x_est = x_interp[location]
        x_dists = x_est - t_series_x
        dists_all = numpy.absolute(x_dists)
        dist_sorted = numpy.argsort(dists_all)
        dist_indices = numpy.where((dists_all < dist_r))[0]
        dists_max = dist_r
        if dists_all[dist_sorted][pt_min_r] > dist_r:
            dist_indices = dist_sorted[0:pt_min_r]
            dists_max = numpy.max(dists_all[dist_indices])
        x_reg = x_dists[dist_indices]
        z_reg = t_series_z[dist_indices]
        dists = dists_all[dist_indices]
        dists_norm = dists / dists_max
        ones = numpy.ones(len(dist_indices))
        if factor == 1:
            A = numpy.transpose(numpy.reshape(numpy.array([(ones), (x_reg)]), (2, -1)))
            w = ((1.0 - (dists_norm**3.0))**3.0)
            w = numpy.reshape(w, (-1, 1))
            A = A * numpy.repeat(w, 2, 1)
        if factor == 2:
            A = numpy.transpose(numpy.reshape(numpy.array([(ones), (x_reg**2.0), (x_reg)]), (3, -1)))
            w = ((1.0 - (dists_norm**3.0))**3.0)
            w = numpy.reshape(w, (-1, 1))
            A = A * numpy.repeat(w, 3, 1)
        b = numpy.reshape(w[:, 0] * z_reg, (-1, 1))
        new_x = numpy.linalg.lstsq(A, b, rcond=None)[0]
        new_z = numpy.reshape(numpy.array([new_x[0]]), (-1, 1))
        output[location] = float(new_z)
        sys.stdout.write(f"\rLOESS Regression {location+1} of {len(output)} complete:           ")
    sys.stdout.write("\rLOESS Regression complete                     ")
    print("")
    return output

today = str(task_id) + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
dataprep_folder = os.path.join(wkspc_DataPrep, today)
analysis_folder = os.path.join(wkspc_Analysis, today)
output_folder = os.path.join(wkspc_Output, today)
os.makedirs(analysis_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
print ("go1")
files_to_copy = ["X_beach_SH8.csv", "date_list.csv", "tide_output.csv", "wave_output.csv"]
for file in files_to_copy:
	shutil.copy(os.path.join(wkspc_DataPrep, file), analysis_folder)

files_to_copy = ["Analysis_DataPrep_Storm_20240117.py", "cmep_xbeach.py", "loess2D.py", "plot_profile.py"]
for file in files_to_copy:
	shutil.copy(os.path.join(wkspc_Analysis, file), analysis_folder)

os.chdir(wkspc)
os.chdir(analysis_folder)
analysis_dataprep_script = "Analysis_DataPrep_Storm_20240117.py"
os.system(f"python {analysis_dataprep_script} {task_id}")
print ("go2")
xbeach_executable = "D:/Dropbox/Rutgers_Geo/2023_NorthBeach/01_Analysis/01_X_beach/XBeach_amarel_test/01_Analysis/xbeach.exe"#"/home/wjs107/xBeach/bin/xbeach"
os.system(xbeach_executable)
print ("go3")
os.chdir(analysis_folder)
plot_script = "plot_profile.py"
os.system(f"python {plot_script} {today}")
print ("go4")
os.chdir(wkspc)
files_to_copy = [
	"jonswap_table.txt", "params.txt", "tide.txt", "wave_runup.txt",
	"Wave_Tide_data.png", "xboutput.nc", "model_profiles.txt", "Geomorph_change.png",
	"Wave_RunUp.png"
] + glob.glob("nh_series*.bcf")
for file in files_to_copy:
	shutil.copy(os.path.join(analysis_folder, file), output_folder)

output_wave_tmp = numpy.genfromtxt(output_folder + "/jonswap_table.txt", delimiter=" ")
output_tide_tmp = numpy.genfromtxt(output_folder + "/tide.txt", delimiter=" ")
output_runup_tmp1 = numpy.genfromtxt(output_folder + "/wave_runup.txt", delimiter=",")
output_geomorph_tmp1 = numpy.genfromtxt(output_folder + "/model_profiles.txt", delimiter=",")
output_geomorph_tmp = numpy.zeros((1, len(output_geomorph_tmp1[0, 1:])))

dist_r1 = 3600.0
pt_min_r = 10
output_runup_tmp = loess(
	output_tide_tmp[:, 0],
	output_runup_tmp1[:, 0],
	output_runup_tmp1[:, 1],
	pt_min_r,
	dist_r1,
	2
)

geo_x = output_geomorph_tmp1[:, 0]
for geo_val in range(len(output_geomorph_tmp1[0, 1:])):
	idx = int(geo_val + 1)
	if geo_val == 0:
		y1 = output_geomorph_tmp1[:, idx]
		integral_y1 = simps(y1[y1 > 0], geo_x[y1 > 0])
		output_geomorph_tmp[0, geo_val] = integral_y1
	if geo_val > 0:
		y2 = output_geomorph_tmp1[:, idx]
		integral_y2 = simps(y2[y2 > 0], geo_x[y2 > 0])
		output_geomorph_tmp[0, geo_val] = integral_y2

output_wave = numpy.zeros((1, len(output_wave_tmp[:, 0])))
output_tide = numpy.zeros((1, len(output_tide_tmp[:, 1])))
output_runup = numpy.zeros((1, len(output_runup_tmp)))
output_geomorph = numpy.zeros((1, len(output_geomorph_tmp[0, :])))

output_wave[0, :] = output_wave_tmp[:, 0]
output_tide[0, :] = output_tide_tmp[:, 1]
output_runup[0, :] = output_runup_tmp
output_geomorph[0, :] = output_geomorph_tmp[0, :]

numpy.savetxt('output_wave_48h_' + str(today) + '.csv', output_wave, delimiter=',', fmt='%4.2f')
numpy.savetxt('output_tide_48h_' + str(today) + '.csv', output_tide, delimiter=',', fmt='%4.2f')
numpy.savetxt('output_runup_48h_' + str(today) + '.csv', output_runup, delimiter=',', fmt='%4.2f')
numpy.savetxt('output_geomorph_48h_' + str(today) + '.csv', output_geomorph, delimiter=',', fmt='%4.2f')

cross_plot_script = "cross_plots.py"
os.system(f"python {cross_plot_script} {task_id} {today}")

finish_time = datetime.datetime.now()
print("Time finished:", time.strftime("%H:%M:%S"))
elapsed_time = finish_time - start_time
hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed_str = f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
print("Elapsed time:", elapsed_str)