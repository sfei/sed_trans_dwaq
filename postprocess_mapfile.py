
# export PYTHONPATH=/opt/software/rusty/stompy/newest_commit/stompy

import os,sys
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
from stompy.model.delft import io as dio
from stompy.model.delft import waq_scenario

outfolder = 'Figures'
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
moviefolder = os.path.join(outfolder,'Concentration_Movie')
if not os.path.exists(moviefolder):
    os.makedirs(moviefolder)



hydro_path = '/fortcollinsvol2/hpcshared/sed_trans/whole_bay_model/hydro/wy2016-fetchwaves/runs/wy2016-fetchwaves/DFM_DELWAQ_wy2016-fetchwaves_bound_temp_salt/wy2016-fetchwaves.hyd'
hydro=waq_scenario.HydroFiles(hyd_path=hydro_path,enable_write_symlink=True)
hydro.infer_2d_elements()
hydro.infer_2d_links()
g = hydro.grid()
map_file = os.path.join('sfbay_dynamo000.map') 
mdata  = dio.read_map(map_file,hydro)

################ First plot the change in sediment bed concentration ########################

# time 0 and time N bed concentrations, only the bottom layer has real concentrations, other layers are dummy values
t0 = str(mdata.time.values[0])[0:10]
tN = str(mdata.time.values[-1])[0:10]
IM1S1 = mdata['IM1S1'][:,-1,:].values.copy()
IM2S1 = mdata['IM2S1'][:,-1,:].values.copy()
IM3S1 = mdata['IM3S1'][:,-1,:].values.copy()
IM1S1[IM1S1<0] = np.nan
IM2S1[IM2S1<0] = np.nan
IM3S1[IM3S1<0] = np.nan
IM1S1_0 = IM1S1[0,:]
IM1S1_N = IM1S1[-1,:]
IM1S1_D = IM1S1_N - IM1S1_0
IM2S1_0 = IM2S1[0,:]
IM2S1_N = IM2S1[-1,:]
IM2S1_D = IM2S1_N - IM2S1_0
IM3S1_0 = IM3S1[0,:]
IM3S1_N = IM3S1[-1,:]
IM3S1_D = IM3S1_N - IM3S1_0
IM1S1_max = np.nanmax(IM1S1_0)
IM2S1_max = np.nanmax(IM2S1_0)
IM3S1_max = np.nanmax(IM3S1_0)
diff_frac = 0.03

# plot the initial and final sediment bed concentrations and the difference
fig, ax = plt.subplots(3,3,figsize=(16,12),constrained_layout=True)
ax = ax.flatten()
ax1 = ax[0]
h = g.plot_cells(values=IM1S1_0, ax=ax1, cmap='turbo')
h.set_clim([0,IM1S1_max*(1+diff_frac)])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('IM1S1 at %s' % t0)
ax1.axis('off')
ax1 = ax[1]
h = g.plot_cells(values=IM1S1_N, ax=ax1, cmap='turbo')
h.set_clim([0,IM1S1_max*(1+diff_frac)])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('IM1S1 at %s' % tN)
ax1.axis('off')
ax1 = ax[2]
h = g.plot_cells(values=IM1S1_D, ax=ax1, cmap='turbo')
h.set_clim([-diff_frac*IM1S1_max,diff_frac*IM1S1_max])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('Change in IM1S1 from\n%s to %s' % (t0,tN))
ax1.axis('off')


ax1 = ax[3]
h = g.plot_cells(values=IM2S1_0, ax=ax1, cmap='turbo')
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
h.set_clim([0,IM2S1_max*(1+diff_frac)])
ax1.set_title('IM2S1 at %s' % t0)
ax1.axis('off')
ax1 = ax[4]
h = g.plot_cells(values=IM2S1_N, ax=ax1, cmap='turbo')
h.set_clim([0,IM2S1_max*(1+diff_frac)])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('IM2S1 at %s' % tN)
ax1.axis('off')
ax1 = ax[5]
h = g.plot_cells(values=IM2S1_D, ax=ax1, cmap='turbo')
h.set_clim([-diff_frac*IM2S1_max,diff_frac*IM2S1_max])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('Change in IM2S1 from\n%s to %s' % (t0,tN))
ax1.axis('off')

ax1 = ax[6]
h = g.plot_cells(values=IM3S1_0, ax=ax1, cmap='turbo')
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
h.set_clim([0,IM3S1_max*(1+diff_frac)])
ax1.set_title('IM3S1 at %s' % t0)
ax1.axis('off')
ax1 = ax[7]
h = g.plot_cells(values=IM3S1_N, ax=ax1, cmap='turbo')
h.set_clim([0,IM3S1_max*(1+diff_frac)])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('IM3S1 at %s' % tN)
ax1.axis('off')
ax1 = ax[8]
h = g.plot_cells(values=IM3S1_D, ax=ax1, cmap='turbo')
h.set_clim([-diff_frac*IM3S1_max,diff_frac*IM3S1_max])
fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
ax1.set_title('Change in IM3S1 from\n%s to %s' % (t0,tN))
ax1.axis('off')

fig.savefig(os.path.join(outfolder,'change_in_bed_sediment.png'))

plt.close('all')

############################ then plot the top and bottom sediment concentration as a movie ############

time = mdata.time.values
ntime = len(time)
IM1_top = mdata['IM1'][:,0,:].values.copy()
IM1_bot = mdata['IM1'][:,-1,:].values.copy()
IM2_top = mdata['IM2'][:,0,:].values.copy()
IM2_bot = mdata['IM2'][:,-1,:].values.copy()
IM3_top = mdata['IM3'][:,0,:].values.copy()
IM3_bot = mdata['IM3'][:,-1,:].values.copy() 
IM1_top[IM1_top<-1.0] = np.nan
IM1_bot[IM1_bot<-1.0] = np.nan
IM2_top[IM2_top<-1.0] = np.nan
IM2_bot[IM2_bot<-1.0] = np.nan
IM3_top[IM3_top<-1.0] = np.nan
IM3_bot[IM3_bot<-1.0] = np.nan
IM1_max = np.max([np.nanpercentile(IM1_top,95),np.nanpercentile(IM1_bot,95)])
IM2_max = np.max([np.nanpercentile(IM2_top,95),np.nanpercentile(IM2_bot,95)])
IM3_max = np.max([np.nanpercentile(IM3_top,95),np.nanpercentile(IM3_bot,95)])
for it in range(ntime):

    fig, ax = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    ax = ax.flatten()

    ax1 = ax[0]
    h = g.plot_cells(values=IM1_top[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM1_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Surface IM1 (mg/L)')
    ax1.axis('off')

    ax1 = ax[1]
    h = g.plot_cells(values=IM2_top[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM2_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Surface IM2 (mg/L)')
    ax1.axis('off')

    ax1 = ax[2]
    h = g.plot_cells(values=IM3_top[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM3_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Surface IM3 (mg/L)')
    ax1.axis('off')


    
    ax1 = ax[3]
    h = g.plot_cells(values=IM1_bot[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM1_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Bottom IM1 (mg/L)')
    ax1.axis('off')

    ax1 = ax[4]
    h = g.plot_cells(values=IM2_bot[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM2_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Bottom IM2 (mg/L)')
    ax1.axis('off')

    ax1 = ax[5]
    h = g.plot_cells(values=IM3_bot[it,:], ax=ax1, cmap='turbo')
    h.set_clim([0,IM3_max])
    fig.colorbar(h, ax=ax1, fraction=0.02,pad=0.01)
    ax1.set_title('Bottom IM3 (mg/L)')
    ax1.axis('off')

    fig.suptitle(time[it])

    fig.savefig(os.path.join(moviefolder,'sediment_conc_movie_%06d.png' % it))

    plt.close('all')