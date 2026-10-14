
# export PYTHONPATH=/opt/software/rusty/stompy/newest_commit/stompy

import os,sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt
from stompy.model.delft import io as dio

# set a start date to skip spinup period
time_start = '2015-10-01'
time_stop = '2016-10-01'

try:
    hdata = xr.open_dataset('dwaq_hist.nc')
except:
    hdata = dio.his_file_xarray("sfbay_dynamo000.his")
    hdata.to_netcdf('dwaq_hist.nc')
hdata = hdata.sel(time=slice(time_start,time_stop))
time = hdata.time.values
ntime = len(time)
layer_depth_nondim = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
nlay = len(layer_depth_nondim)

outfolder = 'Figures'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
figfolder = os.path.join(outfolder,'Station_Plots')
if not os.path.exists(figfolder):
    os.makedirs(figfolder)


data_folder = '/fortcollinsvol2/hpcshared/sed_trans/whole_bay_model/wave_sediment_data/Data_SSC_Processed'

grid_path = '/fortcollinsvol2/hpcshared/sed_trans/whole_bay_model/hydro/wy2016-fetchwaves/runs/wy2016-fetchwaves/DFM_DELWAQ_wy2016-fetchwaves_bound_temp_salt/wy2016-fetchwaves_waqgeom.nc'

station_coords = 'stations_v3_plus_wave_sed.csv'

# read grid and staiton coords
grid = xr.open_dataset(grid_path)
df_coords = pd.read_csv(station_coords, index_col=0)

def ssc2kd(ssc):

    # regression based on peterson sites 10-36
    # https://docs.google.com/presentation/d/13jJgsC3d6nVkD9Q_Xc96ZW35TowQdqp1xQk4iBOe3Dw/edit?slide=id.ge77587f350_0_0#slide=id.ge77587f350_0_0
    kd = 0.298*ssc**0.616

    return kd

def get_station_depth(station):
    df_station = df_coords.loc[df_coords['Station']==station]
    x = df_station.utm_x.values[0]
    y = df_station.utm_y.values[0]
    d2 = (x - grid.FlowElem_xcc.values)**2 + (y - grid.FlowElem_ycc.values)**2
    ind = np.argmin(d2)
    bed_depth = grid.FlowElem_bl.values[ind]
    return bed_depth

def get_station_data(station):

    # get the bed depth
    bed_depth = get_station_depth(station)

    # get the water depth
    water_depth = hdata.sel(region='%s_0' % station).sel(field='TotalDepth').bal.values

    # compute the water surface elevation w.r.t. NAVD88 which I'm assuming is basically the same as MLLW
    water_surface = water_depth + bed_depth

    # depth (meters above bottom)
    depth_mab = np.tile(layer_depth_nondim[-1::-1],(ntime,1))*(np.tile(water_depth,(nlay,1)).transpose())

    # depth (meters below surface)
    depth_mbs = (np.tile(water_depth,(nlay,1)).transpose()) - depth_mab

    # depth (meters below MLLW)
    depth_mbmllw = - ((np.tile(water_surface,(nlay,1)).transpose()) - depth_mbs)

    # concentrations
    IM1 = np.zeros((ntime,nlay))
    IM2 = np.zeros((ntime,nlay))
    IM3 = np.zeros((ntime,nlay))
    for ilay in range(nlay):
        IM1[:,ilay] = hdata.sel(region='%s_%d' % (station,ilay)).sel(field='IM1').bal.values
        IM2[:,ilay] = hdata.sel(region='%s_%d' % (station,ilay)).sel(field='IM2').bal.values
        IM3[:,ilay] = hdata.sel(region='%s_%d' % (station,ilay)).sel(field='IM3').bal.values

    return IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw 

def interp_IM(depthi, depth, IM1, IM2, IM3):

    IM1_interp = np.zeros(ntime)
    IM2_interp = np.zeros(ntime)
    IM3_interp = np.zeros(ntime)
    for itime in range(ntime):
        IM1_interp[itime] = np.interp(depthi, depth[itime,:], IM1[itime,:])
        IM2_interp[itime] = np.interp(depthi, depth[itime,:], IM2[itime,:])
        IM3_interp[itime] = np.interp(depthi, depth[itime,:], IM3[itime,:])

    return IM1_interp, IM2_interp, IM3_interp

def get_observed_ssc(station, fname):

    # read the station data
    df = pd.read_csv(os.path.join(data_folder,station,fname),index_col=0)

    # convert time to utc and select data within simulation period
    df['ts_utc'] = df['ts_pst'].astype('datetime64[ns]') + np.timedelta64(7,'h')
    ind = np.logical_and(df['ts_utc']>=time[0], df['ts_utc']<=time[-1])

    # if there is data within the simulation period, proceed
    if any(ind):
        df = df.loc[ind]

        # add nan's where there are large gaps in the data, then interpolate onto model time axis
        df1 = df.copy(deep=True)
        for irow in range(len(df)-1):
            dt = df.iloc[irow+1]['ts_utc'] - df.iloc[irow]['ts_utc']
            if dt>np.timedelta64(3,'h'):
                df1.loc[len(df1)] = {'ts_utc' : df.iloc[irow]['ts_utc']+np.timedelta64(30,'m'), 'ssc_mg_L' : np.nan}
        df1 = df1.sort_values(by='ts_utc')
        ssc_obs = np.interp((time-time[0])/np.timedelta64(1,'s'), 
                            (df1['ts_utc'].values-time[0])/np.timedelta64(1,'s'), 
                            df1['ssc_mg_L'].values)

    # otherwise return an empty list
    else:
        ssc_obs = []

    return ssc_obs

def make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname):

    # add
    IM_interp = IM1_interp + IM2_interp + IM3_interp

    # plot and save
    fig, ax = plt.subplots(2,1,figsize=(11,8.5),constrained_layout=True)
    ax[0].stackplot(time,IM3_interp,IM2_interp,IM1_interp,labels=['model IM1','model IM2','model IM3'],colors=['yellow','cyan','m'])
    #ax[0].plot(time,IM_interp,'m',label='model IM1+IM2+IM3',alpha=0.5)
    #ax[0].plot(time,IM1_interp,'m--',label='model IM1',alpha=0.5)
    #ax[0].plot(time,IM2_interp,'m-.',label='model IM2',alpha=0.5)
    #ax[0].plot(time,IM3_interp,'m:',label='model IM3',alpha=0.5)
    ax[0].plot(time,ssc_obs,'k',label='observation',alpha=0.5)
    ax[0].legend()
    ax[0].set_ylabel('SSC (mg/L)')
    ax[0].set_xlim((time[0],time[-1]))
    ax[0].grid()
    ax[1].plot(time,4.6/ssc2kd(IM_interp),'m',label='model',alpha=0.5)
    ax[1].plot(time,4.6/ssc2kd(ssc_obs),'k',label='observation',alpha=0.5)
    ax[1].legend()
    ax[1].set_ylabel('4.6/Kd (m)')
    ax[1].set_xlim((time[0],time[-1]))
    ax[1].grid()
    fig.suptitle(figtitle)
    fig.savefig(os.path.join(figfolder,figname))
    plt.close('all')

check_bottom_IM3 = True
if check_bottom_IM3:
    fig1, ax1 = plt.subplots(2,1,figsize=(11,8.5),constrained_layout=True)

station = 'Alcatraz_Island'
fname = 'SSC_1P63_M_BELOW_MLLW.csv'
mbmllw = 1.63
figtitle = 'Alcatraz Island: 1.63 m below MLLW'
figname = 'Alcatraz_Island.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mbmllw,depth_mbmllw,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])


station = 'Benicia_Bridge'
fname = 'SSC_LOWER_19_FT_FROM_BED.csv'
mab = 5.7912
figtitle = 'Benicia Bridge Lower: 5.79 m above bottom'
figname = 'Benicia_Bridge_Lower.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Benicia_Bridge'
fname = 'SSC_UPPER_71_FT_FROM_BED.csv'
mab = 21.6408
figtitle = 'Benicia Bridge Upper: 21.6 m above bottom'
figname = 'Benicia_Bridge_Upper.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Dumbarton_Bridge'
fname = 'SSC_LOWER_4_FT_FROM_BED.csv'
mab = 1.2192
figtitle = 'Dumbarton Bridge Lower: 1.22 m above bottom'
figname = 'Dumbarton_Bridge_Lower.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Dumbarton_Bridge'
fname = 'SSC_UPPER_25_FT_FROM_BED.csv'
mab = 7.62
figtitle = 'Dumbarton Bridge Upper: 7.62 m above bottom'
figname = 'Dumbarton_Bridge_Upper.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Hayward'
fname = 'SSC_0P45_M_FROM_BED.csv'
mab = 0.45
figtitle = 'Hayward: 0.45 m above bottom'
figname = 'Hayward.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data('HYW') # < model uses different station name
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Mallard_Island'
fname = 'SSC_LOWER_5_FT_FROM_BED.csv'
mab = 1.524
figtitle = 'Mallard Island Lower: 1.52 m above bottom'
figname = 'Mallard_Island_Lower.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Mallard_Island'
fname = 'SSC_UPPER_3P3_FT_FROM_SURFACE.csv'
mbs = 1.00584
figtitle = 'Mallard Island Upper: 1.01 m below surface'
figname = 'Mallard_Island_Upper.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mbs,depth_mbs,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Richmond_Bridge'
fname = 'SSC_LOWER_5_FT_FROM_BED.csv'
mab = 1.524
figtitle = 'Richmond Bridge Lower: 1.52 m above bottom'
figname = 'Richmond_Bridge_Lower.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Richmond_Bridge'
fname = 'SSC_UPPER_30_FT_FROM_BED.csv'
mab = 9.144
figtitle = 'Richmond Bridge Upper: 9.14 m above bottom'
figname = 'Richmond_Bridge_Upper.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'San_Leandro_Marina'
fname = 'SSC_1P3_M_FROM_BED.csv'
mab = 1.3
figtitle = 'San Leandro Marina: 1.3 m above bottom'
figname = 'San_Leandro_Marina.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data('SLM') # < model uses different station name
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'San_Mateo_Bridge'
fname = 'SSC_22P25_M_FROM_BED.csv'
mab = 22.25
figtitle = 'San Mateo Bridge: 22.2 m above bottom'
figname = 'San_Mateo_Bridge.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data(station)
IM1_interp, IM2_interp, IM3_interp = interp_IM(mab,depth_mab,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])

station = 'Shoal'
fname = 'SSC_0P9_M_FROM_SURFACE.csv'
mbs = 0.9
figtitle = 'Shoal: 0.9 m below surface'
figname = 'Shoal.png'
IM1,IM2,IM3,depth_mab,depth_mbs,depth_mbmllw = get_station_data('SHL') # < model uses different station name
IM1_interp, IM2_interp, IM3_interp = interp_IM(mbs,depth_mbs,IM1,IM2,IM3)
ssc_obs = get_observed_ssc(station, fname)
if len(ssc_obs)>0:
    make_figure(IM1_interp, IM2_interp, IM3_interp, ssc_obs, figtitle, figname)
    if check_bottom_IM3:
        ax1[0].plot(time, IM3[:,0], label=figname.split('.')[0])
        ax1[1].plot(time, IM3[:,-1], label=figname.split('.')[0])


if check_bottom_IM3:
    ax1[0].legend()
    ax1[0].set_ylabel('Top IM3 (mg/L)')
    ax1[1].legend()
    ax1[1].set_ylabel('Bottom IM3 (mg/L)')
    fig1.savefig(os.path.join(figfolder,'check_bottom_IM3_for_NaN.png'))