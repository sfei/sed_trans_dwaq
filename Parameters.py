# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:43:11 2018
Modified to include mineralization and benthic fluxes
@author: zhenlinz
Making this file portable so that it can be used in other runs. 
"""
import sys
import numpy as np
import xarray as xr
import pandas as pd
from stompy import utils
from stompy.model.delft import waq_scenario, custom_process
import datetime
import os

PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial

class BayDynamo(waq_scenario.Scenario):

    def init_substances(self):
        subs=super(BayDynamo,self).init_substances()
        self.log.info('BayDynamo: init_substances()')

        subs['Continuity'] = Sub(initial=IC(default=1.0))
        
        subs['IM1'] = Sub(initial=IC(default=0.0))
        subs['IM2'] = Sub(initial=IC(default=0.0))
        subs['IM3'] = Sub(initial=IC(default=0.0))
 
        subs['IM1S2']        = Sub(initial=IC(default=0.0), active=False)
        subs['IM2S2']        = Sub(initial=IC(default=0.0), active=False)
        subs['IM3S2']        = Sub(initial=IC(default=0.0), active=False)
        
        data = xr.open_dataset('initial_conitions_IM1S1_IM2S1_IM3S1.nc')
        
        nlayer=10
        values1 = np.tile(data.IM1S1.values, nlayer)
        values2 = np.tile(data.IM2S1.values, nlayer)
        values3 = np.tile(data.IM3S1.values, nlayer)

        subs['IM1S1'] = Sub(initial=IC(seg_values=values1), active=False) # spatially varying initial condition
        subs['IM2S1'] = Sub(initial=IC(seg_values=values2), active=False) # spatially varying initial condition
        subs['IM3S1'] = Sub(initial=IC(seg_values=values3), active=False) # spatially varying initial condition
        
        return subs

    def init_parameters(self):       

        """ exactly the parameters from baybloom.inp """
        params=super(BayDynamo,self).init_parameters()
        self.log.info("Start of BayDynamo parameter defs")
        
        """ ****************************
        Turning on relevant processes
        ********************************
        """
        # from sfei nutrient model
        params["ACTIVE_DynDepth"]=PC(1)
        params["ACTIVE_TotDepth"]=PC(1)
        params["ACTIVE_VertDisp"]=PC(1)
        params["ACTIVE_Veloc"]=PC(1) 

        # Sedimentation IM1
        params['ACTIVE_Sed_IM1']=PC(1) 
        params['ACTIVE_Sed_IM2']=PC(1)
        params['ACTIVE_Sed_IM3']=PC(1)           

        # resuspension old names replaced by S12TraIM1 etc.
        #params['ACTIVE_Res_IM1']=PC(1) 
        #params['ACTIVE_Res_IM2']=PC(1)  
        #params['ACTIVE_Res_IM3']=PC(1) 
        params['ACTIVE_S12TraIM1']=PC(1) 
        params['ACTIVE_S12TraIM2']=PC(1)
        params['ACTIVE_S12TraIM3']=PC(1)  


        # these are needed for S12TraIM1 etc to work
        params['ACTIVE_Res_Pickup']=PC(1) # van Rijn pickup in layer 2
        params['ACTIVE_Res_DM']=PC(1)
        params['ACTIVE_S1_Comp']=PC(1)
        params['ACTIVE_S2_Comp']=PC(1)
        params['ACTIVE_Compos']=PC(1)
              
        return params
