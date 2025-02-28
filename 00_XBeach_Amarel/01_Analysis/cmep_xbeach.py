import numpy as np
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from datetime import datetime
def PressureToLiquidLevel(Pressure):
    """
    Converts pressure (dbar) to a depth (m)
    """
    g = 9.81
    rho = 1025
    Pressure *= 10000
    return Pressure/(g*rho)
    
def grab_index(array,value):
    A = np.abs(array-value)
    return np.where(A==A.min())[0][0]
    
def WriteParams(RunTime,nx,min_x,ToeElev,MorphEnabled):
    assert isinstance(MorphEnabled,bool) == True,"Invalid ErosionEnabled setting, must be True or False"
    f1=open('params.txt', 'w+')
    f1.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'+
    '%%% XBeach-G parameter settings input file                                   %%%\n'+
    '%%%                                                                          %%%\n'+
    '%%% date:     15-Jul-2017 17:38:39                                           %%%\n'+
    '%%% Delta-Shell XBeach plugin                                                %%%\n'+
    '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'+
    '\n'+
    '\n'+
    '%%% Physical processes %%%\n'+
    '\n')
    
    if MorphEnabled == True:
        f1.write('morphology   = 1\n'+
    'sedtrans   = 1\n'+
    '%nonh = 1\n' +
    'swave = 0\n' +
    'wavemodel = nonh\n')
    else:
        f1.write('morphology   = 0\n'+
    'sedtrans   = 0\n'+
    'nonh = 1\n' +
    'swave = 0\n' +
    'wavemodel = surfbeat\n')
    
    f1.write('\n%%% Grid parameters %%%\n' +

    'xfile   = x.grd\n' +
    'yfile   = y.grd\n' +
    'nx   = ' + str(nx) +'\n' +
    'depfile   = bed.dep\n' +

    '\n%%% Model time %%%\n' +

    'tstop   = ' + str(RunTime) + '\n' +

    '\n%%% Tide boundary conditions %%%\n' +

    'zs0file    = tide.txt\n' +

    '\n%%% Wave conditions %%%\n' +

    'bcfile     = jonswap_table.txt\n' +

    '\n%%% Output variables %%%\n' +

    'outputformat = netcdf\n' +
    
    '\n% Runup gauges %\n' +
    'tintp   = 0.5\n' +
    'nrugauge  = 1\n' +
    str(min_x) + ' 0\n')
    f1.close()
    
    if MorphEnabled == True:
        f1 = open('params.txt','a')
        f1.write('\n\n%%% Non-erodible layer %%%\n%%%struct = 1\n' +
        '%%%ne_layer = ne_layer.dep\n')
        f1.write('\n%%% Global Variables %%%\n' +
        'nglobalvar = 1\n' +
        'zb\n\n' +
        'tintg= ' + str(3600.0) + '\n')
        f1.close()
        
    elif MorphEnabled == False:
        f1 = open('params.txt','a')
        f1.write('nglobalvar=0\n')
        f1.close()
    
    f1 = open('params.txt','a')
    f1.write('\n\n%%% Please do not change anything below here %%%\n' +
    'instat     = jons_table\n' +
    'nmeanvar=0\n'+
    
    'thetamin = -90\n' +
    'thetamax = 90\n' +
    'dtheta = 10\n' +
    'D50 =.0004\n' +
    'D90 =.0005\n' +
    'tideloc    = 1\n' +
    'ny   = 0\n' +
    'vardx   = 1\n' +
    'posdwn   = -1\n')
    f1.close()
    return
    
def disper(w, h, g=9.81):
    if not type(w) == list:
        w=[w]
    if not type(h) == list:
        h=[h]
    w2 = [iw**2 * ih / g for (iw,ih) in zip(w,h)]
    q = [iw2 / (1 - np.exp(-(iw2**(5/4))))**(2/5) for iw2 in w2]
    
    thq = np.tanh(q)
    thq2 = 1 - thq**2
    
    a = (1 - q * thq) * thq2
    b = thq + q * thq2
    c = q * thq - w2
    
    D = b**2 - 4 * a * c
    arg = (-b + np.sqrt(D)) / (2 * a)
    iq = np.where(D < 0)[0]
    if iq:
        print(iq)
        arg[iq] = -c[iq] / b[iq]
    q = q + arg
    
    k = np.sign(w) * q / h
    if np.isnan(k).any():
        k = np.array(k)
        k[np.isnan(k)] = 0
    return k
    
class Grid(object):

    global __init__
    global xb_grid_xgrid
    def __init__(self, xin=None, zin=None,**kwargs):#, xin, zin
        self.OPT = {'xgrid' : [],                # predefined xgrid vector
               'Tm' : 5,                    # incident short wave period if you impose time series of wave conditions use the min(Tm) as input
               'dxmin' : 1,                 # minimum required cross shore grid size (usually over land)
               'dxmax' : 10,      # user-specified maximum grid size, when usual wave period / CFL condition does not suffice
               'vardx' : True,              # False = constant dx, True = varying dx
               'g' : 9.81,                  # gravitational constant
               'CFL' : 0.9,                 # Courant number
               'dtref' : 4,                 # Ref value for dt in computing dx from CFL
               'maxfac' : 1.15,             # Maximum allowed grid size ratio
               'wl' : 0,                    # Water level elevation used to estimate water depth
               'depthfac': 2,               # Maximum gridsize to depth ratio
               'ppwl' : 25,                 # desired points per wavelength
               'nonh' : True,              # setting grid to solve individual short waves instead of infragravity waves
               'dxdry' : [],                # grid size to use for dry cells
               'zdry' : [],                 # vertical level above which cells should be considered dry
               'xdry' : [],                 # horizontal (cross-shore) level from which cells should be considered dry
               }
        if kwargs is not None:
            for key, value in kwargs.items():
                self.OPT[key] = value
    
    def xb_grid_xgrid(self,xin, zin):

        if not self.OPT['dxdry']:
            self.OPT['dxdry'] = self.OPT['dxmin']
        if not self.OPT['zdry']:
            self.OPT['zdry'] = self.OPT['wl']
        
        # remove nan values from zin
        xin, zin = np.array(xin), np.array(zin)
        xin, zin = xin[np.isfinite(zin)], zin[np.isfinite(zin)]
        
        # set boundaries
        xend = xin[-1]
        xstart = xin[0]
        xlast = xstart
        
        if not self.OPT['vardx']:
            #TODO
            pass
        elif self.OPT['vardx'] and len(self.OPT['xgrid']):
            #TODO
            pass
        elif self.OPT['vardx']:
        
            # prepare
            hin = [max(self.OPT['wl']-z, 0.01) for z in zin] #-z
            xin2, indices = np.unique(xin, return_index=True)
            fhgr = interp1d(xin2, np.array(hin)[indices])
            fzgr = interp1d(xin2, np.array(zin)[indices])
        
            if self.OPT['nonh']:
                k = disper(2 * np.pi / self.OPT['Tm'], np.max(hin), self.OPT['g'])
            else:
                k = disper(np.pi / (2 * self.OPT['Tm']), np.max(hin), self.OPT['g']) # assume Tlong = 4 * Tshort, instead of Llong = 4*Lshort
            Llong = 2 * np.pi / k
        
            # grid settings
            ii = 0
            xgr = [xstart]
            zgr = [zin[0]]
            hgr = [hin[0]]
            dx = []
            while xlast < xend:
                # minimum grid size in the area
                if len(self.OPT['xdry']):
                    drycell = ii > self.OPT['xdry']
                else:
                    drycell = zgr[ii] > self.OPT['zdry']
                if drycell:
                    localmin = self.OPT['dxdry']
                else:
                    localmin = self.OPT['dxmin']
        
                # compute dx, minimum value dx (on dry land) = dxmin
                dxmax = min(Llong / self.OPT['ppwl'], self.OPT['dxmax'])
                dx.append(np.sqrt(self.OPT['g'] * hgr[ii]) * self.OPT['dtref'] / self.OPT['CFL'])
                dx[ii] = min(dx[ii], self.OPT['depthfac'] * hgr[ii])
                dx[ii] = max(dx[ii], localmin)
                if dxmax > localmin:
                    dx[ii] = min(dx[ii], dxmax)
                else:
                    dx[ii] = localmin
                    if ii == 0:
                        print('Computed dxmax (= ' + str(dxmax) + ' m) is smaller than the user defined dxmin (= ' + str(localmin) + ' m).\n'
                                'Grid will be generated using constant dx = dxmin.\nPlease change dxmin if this is not desired.')
                #
                if ii > 0:
                    if dx[ii] >= self.OPT['maxfac'] * dx[ii - 1]:
                        dx[ii] = self.OPT['maxfac'] * dx[ii - 1]
                    if dx[ii] <= 1. / self.OPT['maxfac'] * dx[ii - 1]:
                        dx[ii] = 1. / self.OPT['maxfac'] * dx[ii - 1]
        
                #
                ii += 1
                xgr.append(float(xgr[ii - 1] + dx[ii - 1]))
                xtemp = min(xgr[ii], xend)
                hgr.append(float(fhgr(xtemp)))
                zgr.append(float(fzgr(xtemp)))
                xlast = xgr[ii]
        
            #print('Optimize cross-shore grid using CFL condition')
        
        else:
            print('vardx must be either True or False')
        
        return xgr, zgr
    
def xbeach_1dprofile(X,Z):
    grid  = Grid()
    __init__(grid)
    xgr,zgr = xb_grid_xgrid(grid,X,Z)
    xgr = np.asarray(xgr)
    zgr = np.asarray(zgr)
    #np.savetxt(output_x,xgr,delimiter=' ')
    #np.savetxt(output_z,zgr,delimiter=' ')
    return xgr,zgr

    
def xbeach_grid_bed_setup(Filename,Dsf,Slope,ToeElev,MorphologyEnabled,CliffedBeach):
    ### Please don't change anything below here ###
    assert isinstance(MorphologyEnabled,bool) == True,"Invalid ErosionEnabled setting, must be True or False"
    assert isinstance(CliffedBeach,bool) == True,"Invalid CliffedBeach setting, must be True or False"
    #read in the X,Y,Z co-ordinates
    X,Y,Z = np.loadtxt(Filename,delimiter=',',unpack=True) 
    assert Z[0] < Z[-1],"Ensure file starts with the offshore value" 
    XBX = np.zeros(len(X))

    #loop through the co-ordinates and calculate the cross-shore distance (X) in the XBX array
    for i in range(1,len(X)): 
        dX = abs(X[i] - X[i-1])
        dY = abs(Y[i] - Y[i-1])
        XBX[i] = np.hypot(dX,dY)
    XBX = np.cumsum(XBX)
    XBZ = Z
    assert XBX[-1] > XBX[0],"X co-ordinates are not orientated as increasing shorewards"
    if Dsf != None:
        assert Dsf < 0,"Dsf must be below 0 m"
    if Slope != None and Slope > 0 and Dsf < 0:
        #works out how far offshore to extend the profile
        dX = np.abs(XBX[-1]-XBX[-2])
        dZ = np.abs(Z[-1]-Z[-2])
        #Slope = dZ/dX #use slope between last two points
        dZ = Dsf-Z[-1]
        AddX = dZ/Slope
        XBX = np.hstack(((XBX[0]+AddX),XBX))
        XBZ = np.hstack((Dsf,Z))
   
    #So we've got the grid (XBX) and bed (XBZ) to extend far enough offshore. The next part of the script
    #ensures there are suffient points along the profile.

    #check for negative zero
    #if XBX[0] == -0:
    #    XBX[0] = 0
        
    #check that the grid and bed start at the offshore, correct if necessary
    #if XBZ[0] > XBZ[-1]:
    #    XBZ = np.flipud(XBZ)
    #if XBX[0] > XBX[-1]:
    #    XBX = np.flipud(XBX)

    #print(XBX[0],XBX[-1])
    #print(XBZ[0],XBZ[-1])
    #XBX -= XBX[0]
        
    XBX_FINAL,XBZ_FINAL = xbeach_1dprofile(XBX,XBZ)
    
    if CliffedBeach == True:
        XBX_FINAL[-1] += 0.25
        XBZ_FINAL[-1] = 10
        
    #XBX_FINAL += np.abs(XBX_FINAL[0])
    XBY_FINAL = np.zeros(len(XBX_FINAL))
    
    np.savetxt('x.grd',XBX_FINAL,fmt='%4.4f')
    np.savetxt('y.grd',XBY_FINAL,fmt='%i')
    np.savetxt('bed.dep',XBZ_FINAL,fmt='%4.4f')
    
    if ToeElev != None and MorphologyEnabled==True:
        NE_LAYER = np.zeros(len(XBX_FINAL))
        max_ind = np.argmax(XBZ_FINAL)
        i = grab_index(XBZ_FINAL[0:max_ind],ToeElev)
        NE_LAYER[0:i+1] = 10
        np.savetxt('ne_layer.dep',NE_LAYER,fmt='%i')
        return XBX_FINAL,XBY_FINAL,XBZ_FINAL,NE_LAYER
    elif ToeElev == None and MorphologyEnabled==True:
        NE_LAYER = np.zeros(len(XBX_FINAL))
        NE_LAYER[:]=10
        if CliffedBeach == True:
            NE_LAYER[-2:-1] = 0
        np.savetxt('ne_layer.dep',NE_LAYER,fmt='%i')
        return XBX_FINAL,XBY_FINAL,XBZ_FINAL,NE_LAYER
    else:
        return XBX_FINAL,XBY_FINAL,XBZ_FINAL,ToeElev
    
        
def WriteWaveFiles(WaveFile,Duration):
    WaveFile = 'waves.txt'
    Duration = 3600
    Hs,Tp = np.loadtxt(WaveFile,delimiter=',',unpack=True)
    for i in range(0,len(Hs)):
        FileName = 'jonswap' + str(i) + '.txt'
        f = open(FileName,'w')
        f.write('Hm0 = %2.2f\nTp = %2.2f' % (Hs[i],Tp[i]))
        f.close()

    FileName = 'filelist.txt'
    f = open(FileName,'w')
    f.write('FILELIST\n')
    for i in range(0,len(Hs)):
        f.write('%i 0.2 jonswap%i.txt\n' % (Duration,i))
    f.close()
    return
        
def WriteTideFiles(TideFile,RunTime,Interval,AWACS,AWACSDepth):
    Tide = np.loadtxt(TideFile)
    Time = np.arange(0,RunTime+Interval,Interval)
    Data = np.vstack((Time,Tide))
    np.savetxt('tide.txt',np.transpose(Data),fmt='%2.2f')
    return


def XBeachAquadoppSetup(AWACSFile,Frequency,AWACSDepth):
    Hm0,Tp,Pressure = np.loadtxt(AWACSFile,delimiter=',',unpack=True)
    
    #to write out a series of files
    # for i in range(0,len(Hm0)):
        # FileName = 'jonswap' + str(i) + '.txt'
        # f = open(FileName,'w')
        # f.write('Hm0 = %2.2f\nTp = %2.2f' % (Hm0[i],Tp[i]))
        # f.close()

    # FileName = 'filelist.txt'
    # f = open(FileName,'w')
    # f.write('FILELIST\n')
    # for i in range(0,len(Hm0)):
        # f.write('%i 0.2 jonswap%i.txt\n' % (Frequency,i))
    # f.close()
    
    # to write out a jonswap table
    FileName = 'jonswap_table.txt'
    f = open(FileName,'w')
    for i in range(0,len(Hm0)):
        f.write('%2.2f %2.2f 270 3.3 10 %2.0f 1\n' % (Hm0[i],Tp[i],Frequency))
    f.close()
    
    ModelRunTime = (len(Hm0)-1) * 3600
    Tide = AWACSDepth + PressureToLiquidLevel(Pressure)
    Time = np.arange(0,ModelRunTime+Frequency,Frequency)
    Data = np.vstack((Time,Tide))
    np.savetxt('tide.txt',np.transpose(Data),fmt='%2.2f')
    return ModelRunTime,np.min(Tide),np.max(Tide)
    
def XBeachAWACSetup(AWACSFile,Frequency,AWACSDepth):
    Hm0,Tp,AST = np.loadtxt(AWACSFile,delimiter=',',unpack=True)
    
    #to write out a series of files
    # for i in range(0,len(Hm0)):
        # FileName = 'jonswap' + str(i) + '.txt'
        # f = open(FileName,'w')
        # f.write('Hm0 = %2.2f\nTp = %2.2f' % (Hm0[i],Tp[i]))
        # f.close()

    # FileName = 'filelist.txt'
    # f = open(FileName,'w')
    # f.write('FILELIST\n')
    # for i in range(0,len(Hm0)):
        # f.write('%i 0.2 jonswap%i.txt\n' % (Frequency,i))
    # f.close()
    
    # to write out a jonswap table
    FileName = 'jonswap_table.txt'
    f = open(FileName,'w')
    for i in range(0,len(Hm0)):
        f.write('%2.2f %2.2f 270 3.3 10 %2.0f 1\n' % (Hm0[i],Tp[i],Frequency))
    f.close()
    
    ModelRunTime = (len(Hm0)-1) * 3600
    Tide = AWACSDepth + AST
    Time = np.arange(0,ModelRunTime+Frequency,Frequency)
    Data = np.vstack((Time,Tide))
    np.savetxt('tide.txt',np.transpose(Data),fmt='%2.2f')
    return ModelRunTime,np.min(Tide),np.max(Tide)
    
def PointWithinRectangle(XMin,XMax,YMin,YMax,PX,PY):
    #test if within r on X and Y rectangular bounds
    if PX >= XMin and PX <= XMax and PY >= YMin and PY <= YMax:
        return True
    else:
        return False
    
def ExtractWW3(NcFile,StartTime,EndTime,PLat,PLon):
    Vars = Dataset(NcFile)
    Lon = np.asarray(Vars.variables['longitude'])
    Lat = np.asarray(Vars.variables['latitude'])
    Hs = np.asarray(Vars.variables['hs'])
    Tp = 1./np.asarray(Vars.variables['fp'])
    Time = np.asarray(Vars.variables['time'])

    StartTime2 = datetime.strptime(StartTime,'%d/%m/%Y %H:%M')
    StartDate = StartTime2.toordinal()
    AddTime = (StartTime2.hour*3600 + StartTime2.minute*60) / 86400.
    StartDate += AddTime
    StartDate -= 726468 #correct to days since 1990

    EndTime2 = datetime.strptime(EndTime,'%d/%m/%Y %H:%M')
    EndDate = EndTime2.toordinal()
    AddTime = (EndTime2.hour*3600 + EndTime2.minute*60) / 86400.
    EndDate += AddTime
    EndDate -= 726468 #correct to days since 1990
    
    assert StartDate > Time[0],"WaveWatchIII StartDate is before beginning of model output"
    assert StartDate < Time[-1],"WaveWatchIII StartDate is after end of model output"
    assert EndDate > Time[0],"WaveWatchIII EndDate is before beginning of model output"
    assert EndDate < Time[-1],"WaveWatchIII EndDate is after end of model output"
    assert EndDate > StartDate, "WaveWatchIII StartDate is before end date"

    SI = grab_index(Time,StartDate)
    EI = grab_index(Time,EndDate)

    WW3Time = Time[SI:EI+1]

    #check that the user input point is in the model grid
    assert PointWithinRectangle(np.min(Lon),np.max(Lon),np.min(Lat),np.max(Lat),PLon,PLat) == True,"Desired grid point outside WaveWatchIII model domain"
    
    #find the closest point in the grid
    PLonI = grab_index(Lon,PLon)
    PLatI = grab_index(Lat,PLat)

    WW3_Hs = Hs[SI:EI+1,PLatI,PLonI]
    WW3_Tp = Tp[SI:EI+1,PLatI,PLonI]

    f = open('jonswap_table.txt','w')
    for i in range(0,len(WW3_Hs)):
        f.write('%2.2f %2.2f 270 3.3 10 3600 1\n' % (WW3_Hs[i],WW3_Tp[i]))
    f.close()
    return
    
def ExtractNemo(NcFile,StartTime,EndTime,PLat,PLon):
    Vars = Dataset(NcFile)
    Lon = np.asarray(Vars.variables['nav_lon'])
    Lat = np.asarray(Vars.variables['nav_lat'])
    SSH = np.asarray(Vars.variables['zos'])
    Time = np.asarray(Vars.variables['time_counter'])
    Time /= 86400 #convert from seconds to days
    
    RefTime = '01/01/1900 00:00' #NEMO's time datum
    RefTime2 = datetime.strptime(RefTime,'%d/%m/%Y %H:%M')
    RefDate = RefTime2.toordinal()
    
    Time += RefDate #Correct NEMO time to Python datetime to match start/end dates

    StartTime2 = datetime.strptime(StartTime,'%d/%m/%Y %H:%M')
    StartDate = StartTime2.toordinal()
    AddTime = (StartTime2.hour*3600 + StartTime2.minute*60) / 86400.
    StartDate += AddTime
    
    assert StartDate > Time[0],"NEMO StartDate is before beginning of model output"
    assert StartDate < Time[-1],"NEMO StartDate is after end of model output"
    

    EndTime2 = datetime.strptime(EndTime,'%d/%m/%Y %H:%M')
    EndDate = EndTime2.toordinal()
    AddTime = (EndTime2.hour*3600 + EndTime2.minute*60) / 86400.
    EndDate += AddTime
    
    assert EndDate > Time[0],"NEMO EndDate is before beginning of model output"
    assert EndDate < Time[-1],"NEMO EndDate is after end of model output"
    assert EndDate > StartDate, "NEMO StartDate is before end date"
    
    SI = grab_index(Time,StartDate)
    EI = grab_index(Time,EndDate)

    NemoTime = Time[SI:EI+1]

    #check that the user input point is in the model grid
    assert PointWithinRectangle(np.min(Lon),np.max(Lon),np.min(Lat),np.max(Lat),PLon,PLat) == True,"Desired grid point outside NEMO model domain"
    
    PLonI = grab_index(Lon[0,:],PLon)
    PLatI = grab_index(Lat[:,0],PLat)

    NemoSSH = SSH[SI:EI+1,PLatI,PLonI]
    
    ModelRunTime = (len(NemoSSH)-1)*3600
    RunTime = np.arange(0,ModelRunTime+3600,3600)

    f = open('tide.txt','w')
    for i in range(0,len(NemoSSH)):
        f.write('%i %2.2f\n' % (RunTime[i],NemoSSH[i]))
    f.close()
    return ModelRunTime,np.min(NemoSSH),np.max(NemoSSH)
    
def PlotWaveRunUp(OutputFile):
	XBFile = Dataset('xboutput.nc')
	RunUp = np.squeeze(np.asarray(XBFile.variables['point_zs']))
	print(RunUp)
	w1 = np.where(RunUp<1000)[0]
	Time = np.squeeze(np.asarray(XBFile.variables['pointtime'])) #extract bed levels
	Data = np.transpose(np.vstack((Time[w1],RunUp[w1])))
	R0 = np.round(np.max(RunUp[w1]),2)
	R2 = np.round(np.percentile(RunUp[w1],98),2)
	R5 = np.round(np.percentile(RunUp[w1],95),2)
	R10 = np.round(np.percentile(RunUp[w1],90),2)
	print('Maximum wave run-up(m): ',R0)
	print('2% exceedance wave run-up (m): ',R2)
	print('5% exceedance wave run-up (m): ',R5)
	print('10% exceedance wave run-up (m): ',R10)
	np.savetxt(OutputFile,Data,delimiter=',',fmt='%4.2f')
	return RunUp[w1],Time[w1]
    
def PlotStormProfile(OutputFile):
	XBFile = Dataset('xboutput.nc')
	Zb = np.squeeze(np.asarray(XBFile.variables['zb'])) #extract bed levels
	X = np.squeeze(np.asarray(XBFile.variables['globalx'])) #extract cross-shore grid
	#print(XBFile.variables)
	'''
	print(X)
	print(Zb)
	print(Zb[0])
	print(Zb[-1])
	'''
	Data = np.transpose(np.vstack((X,Zb)))
	
	np.savetxt(OutputFile,Data,delimiter=',',fmt='%4.2f')
	return Data