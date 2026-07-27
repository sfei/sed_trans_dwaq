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
        values = np.tile(data.IM1S1.values, nlayer)

        subs['IM1S1'] = Sub(initial=IC(seg_values=values), active=False) # spatially varying initial condition
        subs['IM2S1'] = Sub(initial=IC(seg_values=values), active=False) # spatially varying initial condition
        subs['IM3S1'] = Sub(initial=IC(seg_values=values), active=False) # spatially varying initial condition
        
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

        # from jill and mick model
        params['ACTIVE_Res_Pickup']=PC(0)
        params['ACTIVE_Sed_IM1']=PC(0)
        params['ACTIVE_Res_IM1']=PC(0)
        params['ACTIVE_Sed_IM2']=PC(0)
        params['ACTIVE_Res_IM2']=PC(0)
        params['ACTIVE_Sed_IM3']=PC(0)
        params['ACTIVE_Res_IM3']=PC(0)
        #;CONSTANTS 'ACTIVE_CalVS_IM1' DATA 0
        #;CONSTANTS 'ACTIVE_CalVS_IM2' DATA 0
        params['ACTIVE_DynDepth']=PC(0)
        params['ACTIVE_Res_DM']=PC(0)
        params['ACTIVE_S1_Comp']=PC(0)
        params['ACTIVE_S2_Comp']=PC(0)
        params['ACTIVE_Compos']=PC(0)
        params['ACTIVE_TotDepth']=PC(0)
              
        return params
