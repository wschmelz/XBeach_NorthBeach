import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as numpy
import cmep_xbeach
from cmep_xbeach import PlotStormProfile
from cmep_xbeach import PlotWaveRunUp
import sys

backslash = '\\'
wkspc_analysis = str(os.getcwd()).replace(backslash,"/") + "/"
task_id = sys.argv[1]

OutputFile = 'model_profiles.txt'
Data = PlotStormProfile(OutputFile)
fig = plt.figure(1, figsize=(15, 7.5))
ax1 = plt.subplot(111)
ax1.plot(Data[:, 0], Data[:, 1], 'k', label='Initial Profile')
ax1.plot(Data[:, 0], Data[:, -1], 'r', label='Final Profile')
ax1.set_xlabel('Cross-shore distance (m)')
ax1.set_ylabel('Elevation (m)')
ax1.set_xlim(1350, 1700)
ax1.set_ylim(-2, 3)
ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(2.5))
ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
ax1.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5, alpha=0.5)
print("Data:", Data)
plt.tight_layout()
pltname = wkspc_analysis + "Geomorph_change" + str(task_id) + ".png"
plt.savefig(pltname, dpi=300)
pltname = wkspc_analysis + "Geomorph_change.png"
plt.savefig(pltname, dpi=300)

plt.rc('font', size=20)
OutputFile = 'wave_runup.txt'
fig = plt.figure(2)
RunUp, Time = PlotWaveRunUp(OutputFile)
plt.xlabel('Time (hours)')
plt.ylabel('Wave Run-Up (m)')
plt.xlim([0, Time[-1] / (60 * 60)])
plt.plot(Time / (60 * 60), RunUp, 'k', linewidth=0.15)
plt.tight_layout()
pltname = wkspc_analysis + "Wave_RunUp" + str(task_id) + ".png"
plt.savefig(pltname, dpi=300)
pltname = wkspc_analysis + "Wave_RunUp.png"
plt.savefig(pltname, dpi=300)
plt.close()