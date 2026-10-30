"""
june 2026 allie modified to run sediment transport model, based on
jill and mick's work, in turn based on rachel allen's d3d model 

"""
import os
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np
import xarray as xr
import pandas as pd
import shapely.wkt
from six import iteritems
from importlib import reload

from stompy import utils
from stompy.model.delft import waq_scenario

import Parameters as BayDynamo
reload(BayDynamo)

# set the simulation time period and point to the hydro input files
sim_dates = pd.to_datetime(['2015-08-01 00:00:00','2016-10-01 00:00:00'])  
hydro_path = '/fortcollinsvol2/hpcshared/sed_trans/whole_bay_model/hydro/wy2016-fetchwaves/runs/wy2016-fetchwaves-sigma/DFM_DELWAQ_wy2016-fetchwaves-sigma_bound_temp_salt/wy2016-fetchwaves-sigma.hyd'

# mallard island ssc from lester
df=pd.read_csv('Mallard_v20.csv')
df_mallard = df['SSC (mg/L)']
df_mallard.index = pd.DatetimeIndex(df['time'])
ind = ~np.isnan(df_mallard)
df_mallard = df_mallard.loc[ind]

# tributary inputs from Pedro and Matt (Watershed Dynamic Model)
data_fresh = xr.open_dataset('sfbay_freshwater_OCT2026.nc')


# load the hydro data
hydro=waq_scenario.HydroFiles(hyd_path=hydro_path,enable_write_symlink=True)

PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial

class Scen(BayDynamo.BayDynamo):
    name="sfbay_dynamo000"
    desc=('sfbay_dynamo000',
          'wy2020',
          'fr_grid')
    base_x_dispersion = 0.0 # m**2/s - constant horizontal dispersion coefficient 
    base_y_dispersion = 0.0
    integration_option= """22.62 ; 
    LOWER-ORDER-AT-BOUND NODISP-AT-BOUND 
    BALANCES-OLD-STYLE BALANCES-GPP-STYLE
    BAL_NOLUMPPROCESSES BAL_NOLUMPLOADS BAL_NOLUMPTRANSPORT
    BAL_NOSUPPRESSSPACE BAL_NOSUPPRESSTIME 
    """
    
    _base_path='auto'

    time_step=1000 # dwaq HHMMSS integer format

    map_formats=['binary']
    
    storm_sources=['SCLARAVW2_flow',
                   'SCLARAVW1_flow',
                   'SCLARAVW4_flow',
                   'SCLARAVW3_flow',
                   'UALAMEDA_flow',
                   'EBAYS_flow',
                   'COYOTE_flow',
                   'PENINSULb1_flow',
                   'EBAYCc3_flow',
                   'USANLORZ_flow',
                   'PENINSULb3_flow',
                   'PENINSULb4_flow',
                   'EBAYCc2_flow',
                   'PENINSULb6_flow',
                   'PENINSULb2_flow',
                   'PENINSULb7_flow',
                   'PENINSULb5_flow',
                   'SCLARAVCc_flow',
                   'SCLARAVW5_flow',
                   'MARINS1_flow',
                   'EBAYCc6_flow',
                   'EBAYCc1_flow',
                   'EBAYCc5_flow',
                   'EBAYCc4_flow',
                   'MARINN_flow',
                   'NAPA_flow',
                   'CCOSTAW2_flow',
                   'CCOSTAW3_flow',
                   'MARINS3_flow',
                   'MARINS2_flow',
                   'PETALUMA_flow',
                   'SONOMA_flow',
                   'CCOSTAW1_flow',
                   'SOLANOWc_flow',
                   'CCOSTAC2_flow',
                   'EBAYN1_flow',
                   'EBAYN4_flow',
                   'EBAYN2_flow',
                   'EBAYN3_flow',
                   'SOLANOWa_flow',
                   'SOLANOWb_flow',
                   'CCOSTAC3_flow',
                   'CCOSTAC1_flow',
                   'CCOSTAC4_flow']
    
    delta_sources=['Jersey_flow',
                   'RioVista_flow']

    sea_sources=[ 'Sea_ssh' ]
 
    # The above four sources complete all boundary sources for the run
    all_sources = storm_sources+ delta_sources+sea_sources
        
    
    def init_substances(self):
        subs=super(Scen,self).init_substances()

     
        
        boundaries=self.hydro.boundary_defs()
        allitems = [boundary.decode("utf-8") for boundary in set(boundaries['type'])]        
         
        # continuity tracer
        self.src_tags.append(dict(tracer='continuity',
                                  items=self.all_sources,
                                  value=1.0))
        
        # use lester's spreadsheet for delta sources and partition
        # 12% Big Flocs, 58% Mud, 30% Sand  
        for k in self.delta_sources: 
                
            # mud
            self.src_tags.append(dict(tracer='IM2',
                                      items=k,
                                      value=0.12*df_mallard))
            
            # big flocs
            self.src_tags.append(dict(tracer='IM1',
                                      items=k,
                                      value=0.58*df_mallard))

            # sand
            self.src_tags.append(dict(tracer='IM3',
                                      items=k,
                                      value=0.30*df_mallard)) 

        for k in self.storm_sources:

            station = k.split('_flow')[0]
            df_ssc = pd.DataFrame(columns=['ssc_clay_mg_L','ssc_silt_mg_L','ssc_sand_mg_L'])
            df_ssc['ssc_clay_mg_L'] = data_fresh.sel(station=station)['ssc_clay_mg_L']
            df_ssc['ssc_silt_mg_L'] = data_fresh.sel(station=station)['ssc_silt_mg_L']
            df_ssc['ssc_sand_mg_L'] = data_fresh.sel(station=station)['ssc_sand_mg_L']
            df_ssc.index = pd.DatetimeIndex(data_fresh.time)
            df_ssc.index.name='time'

            # mud
            self.src_tags.append(dict(tracer='IM2',
                                      items=k,
                                      value=df_ssc['ssc_clay_mg_L']))
            
            # big flocs
            self.src_tags.append(dict(tracer='IM1',
                                      items=k,
                                      value=df_ssc['ssc_silt_mg_L']))

            # sand
            self.src_tags.append(dict(tracer='IM3',
                                      items=k,
                                      value=df_ssc['ssc_sand_mg_L'])) 
           
        return subs
    
    def init_parameters(self):

        # choose which processes are enabled.  Includes some
        # parameters which are not currently used.
        params=super(Scen,self).init_parameters()
          
        params['CLOSE_ERR']=PC(1) # ; If defined, allow delwaq to correct water volumes to keep concentrations continuous
        params['maxiter']=PC(100) # ; Maximum number of iterations
        params['tolerance']=PC(1E-07) # ; Convergence tolerance
        params['iteration report']=PC(0) # ; Write iteration report (when 1) or not (when 0)

        # constants from Jill and Mick's sediment transport model
        #;CONSTANTS 'TauShields' DATA 0.1
        #;CONSTANTS 'GRAIN50' DATA 0.0003
        #;CONSTANTS 'GRAV' DATA 9.8
        params['KinViscos']=PC(1E-06)
        params['RHOSAND']=PC(2600000)
        params['RhoWater']=PC(1000) # ; follows from D3D
        params['PORS1']=PC(0.4)
        #;CONSTANTS 'PORS2' DATA 0.4
        #;CONSTANTS 'ThickS2' DATA 0.1
        params['MinDepth']=PC(0.01)
        #;CONSTANTS 'MaxResPup' DATA 1E+20
        #;CONSTANTS 'FactResPup' DATA 1.75E-07
        params['VSedIM1']=PC(86.4) # ; follows from D3D
        params['TaucSIM1']=PC(1000) #  ;follows from D3D
        #;CONSTANTS 'FrIM1SedS2' DATA 0.1
        #;CONSTANTS 'FrTIMS2Max' DATA 1
        params['SWResIM1']=PC(1)
        params['SWResusp']=PC(1)
        params['ZResIM1']=PC(432.0) # ; 10% of D3D
        params['VResIM1']=PC(100) # ; 0.4
        params['TaucRS1IM1']=PC(0.1) # ; follows from d3d
        #;CONSTANTS 'TaucRS2IM1' DATA 1000
        params['VSedIM2']=PC(17.28) # ; follows from d3d CONSTANTS 'nIM2' DATA 0
        params['TaucSIM2']=PC(1000) # ; follows from d3d
        #;CONSTANTS 'FrIM2SedS2' DATA 0.1
        params['SWResIM2']=PC(1)
        params['ZResIM2']=PC(86.400) # ; 10% of D3D
        params['VResIM2']=PC(100) # ; 0.4
        params['TaucRS1IM2']=PC(0.1) # ; follows from d3d
        #;CONSTANTS 'TaucRS2IM2' DATA 1000
        params['psedminIM2']=PC(1.00E-01)
        params['VSedIM3']=PC(2246.4) # ; follows from D3D
        params['TaucSIM3']=PC(1000) #  ;follows from D3D
        params['SWResIM3']=PC(1)
        params['ZResIM3']=PC(86.400) # ; 10% of D3D
        params['VResIM3']=PC(100) # ; 0.4
        params['TaucRS1IM3']=PC(0.16) # ; follows from d3d
        #;CONSTANTS 'SWResDM' DATA 1
        #;CONSTANTS 'ZResDM' DATA 2246.4
        #;CONSTANTS 'VResDM' DATA 0.4
        #;CONSTANTS 'fResS1DM' DATA 86.400

        # comment these out because lsp file says "is not used by the proces system"
        # params['NOTHREADS']=PC(4) # ; Number of threads used by delwaq, equal number of substances
        # params['DRY_THRESH']=PC(0.001) # ; Dry cell threshold
        # params['CrSS']=PC(25)
        # params['nIM1']=PC(0.1)
        # params['Temp']=PC(10)
        # params['Manncoef']=PC(0.025) # ; follows from d3d 
        # params['SwChezy']=PC(2) # ; follows from d3d
        # params['nIM3']=PC(0.1)

        return params
        
    def cmd_default(self):
        
        self.share_path = os.environ.get('DELFT_SHARE')
        self.delwaq2_path = os.environ.get('DELWAQ2_PATH')   
        
        self.cmd_write_hydro()
        self.cmd_write_inp()
        self.cmd_delwaq1()
        self.cmd_delwaq2()        
        #self.cmd_write_nc()        


        
        
    def __init__(self,*a,**k):
        super(Scen,self).__init__(*a,**k)

        extra_fields=('Tau','TotalDepth','Depth','Velocity','TIM', 
                      'FrIM1S2Pup','fRIM2S2Pup','fRIM3S2Pup',
                      'Pshields','FrTIMS2','PSedIM1','fSedIM1',
                      'fSedIM1S2','fResS1IM1','fResS2IM1','PSedIM2','fSedIM2',
                      'fSedIM2S2','FrIM2S1','FrIM2S2','fResS1IM2','fResS2IM2',
                      'PSedIM3','fSedIM3','fSedIM3S2','FrIM3S1','FrIM3S2',
                      'fResS1IM3','fResS2IM3')                                             
                                    
                     
        self.map_output+=extra_fields

        self.hist_output+=extra_fields
        
        #self.stat_output +=extra_fields 

        self.mon_output = ()#('TotalDepth')
        
        DAILY=1000000
        self.map_time_step=6000 # every 60 min. 1000000 # daily
#        self.mon_time_step=10000 # every 100 min.
#        self.hist_time_step=10000 # every 100 min
        self.mon_time_step= 1000000 # daily # 6000 # every 60 min. -- format here mmss - this is not an integer specification of time!!
        self.hist_time_step=1000 # every 10 min        
        #self.add_usgs_transect_monitor() 
        self.add_usgs_monitor_areas()
        #self.add_disp()

 
    def add_usgs_monitor_areas(self):
        
        dfs = pd.read_csv('stations_v3_plus_wave_sed.csv') 
        xy = np.array([dfs['utm_x'],dfs['utm_y']]).transpose()
        
        self.hydro.infer_2d_elements()
        g=self.hydro.grid()
        elt_sel=[g.select_cells_nearest(pnt,inside=False)
                 for pnt in xy]

        segs=[np.nonzero(elt==hydro.seg_to_2d_element)[0]  # output the depth-average
               for elt in elt_sel ]
#        segs=[ [np.squeeze(np.nonzero(elt==hydro.seg_to_2d_element))[0]] # output only the surface level
#               for elt in elt_sel ]
        for i in np.arange(len(segs)):
            self.monitor_areas=self.monitor_areas + ((dfs['Station'].values[i],segs[i]),)
        
        # for the whole depth output
        utm = np.array([dfs['utm_x'],dfs['utm_y']]).transpose()
        Sitename = dfs['Station'].values
        for Sitename_i,utm_i in zip(Sitename,utm):
            elt_sel=g.select_cells_nearest(utm_i,inside=False)
            segs=np.nonzero(elt_sel==hydro.seg_to_2d_element)[0]
            for i in np.arange(len(segs)):
                self.monitor_areas=self.monitor_areas + (('{}_{}'.format(Sitename_i,i),[segs[i]]),)            

sec=datetime.timedelta(seconds=1)
sim_secs = sim_dates - sim_dates[0]
if 0:  # short run for testing:
    start_time=hydro.time0+ sim_secs # hydro.t_secs[ 0]*sec
    # and run for 20 days
    stop_time=start_time + 20*24*3600*sec
    #map_time_step=3000 # half hour
if 1: # long run
    #start_time=hydro.time0+ sim_secs # hydro.t_secs[ 0]*sec
    start_time=hydro.time0+sim_secs[0]
    stop_time=hydro.time0+sim_secs[1]
    #stop_time=start_time + 5*24*3600*sec    
    #map_time_step=60000#1000000 # daily


scen=Scen(hydro=hydro,
      start_time=start_time,
      stop_time=stop_time,
      base_path = os.getcwd(),
      overwrite=True)

scen.cmd_default()



