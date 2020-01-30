# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:09:47 2019

@author: 813146
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import re

def hcpc_mod_splitter(filename):
    
    data = filename
    data = data[['HCPC Code']]
	    
    data['HCPC'] = ''
    data['Modifier'] = ''
    
    def checkToAddComma(length, value):
        if length > 0:
            return ',' + value;
        return value;
    
    for i in range(data.shape[0]):
        hcpcCode = data['HCPC Code'][i]
        if len(hcpcCode) != 0:
            hcpcCodeList = re.split(';' ,hcpcCode.replace('\u00B2',';'))
            if len(hcpcCodeList) == 1 :
               data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[0])
               
            if len( hcpcCodeList) == 2  :
               data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[0])
               data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[1])
               
            if len(hcpcCodeList) > 2 and len(hcpcCodeList) < 5:
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[0])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[1])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[2])
                data['Modifier'][i] +=  checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[3])
                data['HCPC'][i] = ",".join([*np.unique(data['HCPC'][i].split(","))])          
                data['Modifier'][i] = ",".join([*np.unique(data['Modifier'][i].split(","))])
                
            if len(hcpcCodeList) >= 5 and len(hcpcCodeList) < 7:
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[0])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[1])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[2])
                data['Modifier'][i] +=  checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[3])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[4])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[5])
                data['HCPC'][i] = ",".join([*np.unique(data['HCPC'][i].split(","))])          
                data['Modifier'][i] = ",".join([*np.unique(data['Modifier'][i].split(","))])
            
            if len(hcpcCodeList) > 6:
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[0])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[1])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[2])
                data['Modifier'][i] +=  checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[3])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[4])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[5])
                data['HCPC'][i] += checkToAddComma(len(data['HCPC'][i]),hcpcCodeList[6])
                data['Modifier'][i] += checkToAddComma(len(data['Modifier'][i]),hcpcCodeList[7])
                data['HCPC'][i] = ",".join([*np.unique(data['HCPC'][i].split(","))])          
                data['Modifier'][i] = ",".join([*np.unique(data['Modifier'][i].split(","))])
                
    data.drop('HCPC Code', axis = 1, inplace = True)
    
    return data
    

def preprocs(filename):
    
    '''
    To preprocess the raw file as per requirements for ML model.
    
    '''
     
    data = filename
    data = data[['SO ID', 'Patient ID','Clinic ID','SO Dollars','HCPC Code','Claim Type','Insurance ID','Insurance Name','Patient DOB','Patient State', 'Dr. ID','Dx Code']]
    
    data = data[pd.notnull(data['HCPC Code'])]
    data = data[pd.notnull(data['Patient DOB'])]
    data = data[data['SO Dollars'] >= 0]
    data.fillna('~',inplace = True)       
        
    data1 = hcpc_mod_splitter(data)
    data['HCPC'] = data1['HCPC']
    data['Modifier'] = data1['Modifier']
    data['HCPC_dummy'] = data1['HCPC']
    data['Modifier_dummy'] = data1['Modifier']
           
#   
    ##Converting Amount to log to avoid right skew
    data['SO Dollars'] = np.log(data['SO Dollars']+1)      
    
    
    ## Binning HCPC code based on 1 yr data
    val_to_keep_HCPC = ['L4361','L3908','E0114','L3660','L1902','L3809','L3670','L3260','L1833','L1820','L1812,L2795','L3984','L4350','L1830',
'A4565','L3924','L1812','L4387','L3995','A4467','L4397','E0143','L3761','L3265','A9270','L3170','L0642','L0120','E0135',
'E0100','L0174','L3980','A4570','E0218','L1851','L3675','L0641','L3650','L0621','L4360','L4360,L4361','L1906','A6530',
'E0191','L0150','L0172','L1620','A6533','L1810,L1812,L2795','L1930']
    
    data.loc[~data['HCPC'].isin(val_to_keep_HCPC),'HCPC']= "OTHERS"
    
    ## Binning Modifier code based on 1 yr data
    val_to_keep_Mod = ['RT','LT','NU','NURT','NULT','KXRT','KXLT','~','NUKF','RTKX','LTKX',
                   'GZRT','GZLT','NUKX','CGRT','CGLT','CG','RTGZ','LTGZ','GYRT','GYLT',
                   'LTRT','NUGZ','KXRTGZ','KXLTGZ','NUSC','GZ','RTLT','RTGY','LTGY']
    
    data.loc[~data['Modifier'].isin(val_to_keep_Mod),'Modifier']= "OTHERS"
    
    ## Binning Insurance code based on 1 yr data
    val_to_keep_Insur = ['M0003','D3425','M0002','D3622','D3351','M0004','D166','D1677','D107',
                     'D8601','M0001','D233','D3308','D7744','D5371','D186','D225','D125','D3685',
                     'D113','D249','D3982','D218','D6101','D8764','D10194','D2004','D1015','D8314',
                     'D140','D8585','D7188','D6353','D3844','D2952','D168','D382','D2647','D5943',
                     'D135','D10495','D605','D6190','D608','D8570','D8192','D88','D3773','D5446',
                     'D1458','D8435','D8541','D3541','D3302','D866','D6204','D5172','D9607','D238',
                     'D263','D5597','D128','D105','D9624','D10421','D1965','D8867','D9418','D9803',
                     'D6850','D700','D2437','D508','D2187','D3029','D1853','D119','D123','D8096',
                     'D2182','D8301','D138','D9694','D853','D5279','D1272','D11125','D7892','D150',
                     'D9615','D7783','D9144','D339','D5996','D9882','D1551','D8766','D23','D7635']
    
    data.loc[~data['Insurance ID'].isin(val_to_keep_Insur),'Insurance ID']= "OTHERS"
    
    ## Binning Insurance Name based on 1 yr data
    val_to_keep_InsurName = ['MEDICARE REGION C','UNITED HEALTH CHOICE PLUS','AETNA NONHMO','CIGNA','BCBS NC NONHMO',
                     'MEDICARE REGION B','MEDICARE REGION D','BCBS TX NONHMO','ANTHEM IN',
'BS CALIFORNIA FEP','UMR','ANTHEM OH NON HMO','BCBS MI','BCBS TN CLAIMS','UNITED HEALTHCARE','ANTHEM GA','BCBS SC PPO',
                     'BCBS WESTERN NY NONHMO','MEDICARE REGION A',
'BCBS AL','ANTHEM BLUECROSS CA','HUMANA MED ADV','BCBS IL NON HMO','AARP MEDICARE COMPLETE','BCBS OK PPO','HUMANA GOLD',
                     'BS CA O & P NON HMO','AETNA MEDICARE PPO PLAN',
'ANTHEM KY','ANTHEM MO','UHCC LPPO GROUP MED ADV','MED MUTUAL SUPER MED PPO','CARECENTRIX OC A&E CIGNA',
                     'UHCC LPPO MEDICARE ADV','CAREFIRST BCBS MD','OK STATE MEDICAID',
'WELLCARE GA MEDICAID','HUMANA','TRICARE WEST','CARECENTRIX BCBS FL O&P','ANTHEM VA PPO','BCBS TN MEDICAID',
                     'MERIDIAN HLTH IL MEDICAID','BCBS SC STATE HEALTH PLAN',
'PREMERA BC WA BLUECARD','MERIDIAN MI HP MEDICAID','ANTHEM CO','HIGHMARK PA NON HMO','REGENCE BCBS UT NONHMO',
                     'UHCC TN MEDICAID','ANTHEM WI','BCBS KS (TOPEKA)',
'HIGHMARK BCBS OF DE','AMERIGROUP TN MEDICAID','MEDCOST LLC','MANAGED HLTH SERV MG MD','WELLCARE KY MCD',
                     'MOLINA HEALTHCARE WA MEDICAID','AMERIGROUP GA MEDICAID',
'INDEPENDENCE/PERS CHOICE','BCBS MI MED ADV','CARECENTRIX HORIZON O&P','EMPIRE PLAN UNITED HEALTHCARE','AETNA HMO',
                     'WELLMARK BCBS IA','ANTHEM IN HOOSIER HLTHWSE',
'PEACH STATE HEALTH PLAN','ANTHEM IN HEALTHY IN PLAN','CARESOURCE OH','MOLINA HEALTHCARE IL','UHCC COMM OH MEDICAID',
                     'CORESOURCE','UHCC DUAL COMPLETE ME ADV','BCBS LA NON HMO','HIGHMARK HLTH OPTIONS DE']
    
    data.loc[~data['Insurance Name'].isin(val_to_keep_InsurName),'Insurance Name']= "OTHERS"
    
    ##Binning Patient state based on 1 yr data
    val_to_keep_PatState = ['NC','OH','IN','TX','CA','FL','TN','GA','MI','IL','NY',
                        'OK','SC','PA','WA','MO','CO','WI','MD','AL','KY','VA',
                        'NJ','DE','UT','KS','AZ','IA','LA','MN']
    
    data.loc[~data['Patient State'].isin(val_to_keep_PatState),'Patient State']= "OTHERS"
               
    ## Joining SO ID & Patient ID to make it a unique code
    data['Patient_SO_ID'] = ''
    data['Patient_SO_ID'] = data['Patient ID'] + "-" + data['SO ID']
    data.drop(['SO ID','Patient ID'], axis = 1, inplace = True)
    
    # Converting Patient DOB into his Age
        
    try:
        data['Patient DOB'] = data['Patient DOB'].map(lambda x: x.year)
    except:
        pass
    
    try:
        data['Patient DOB'] = data['Patient DOB'].map(lambda x: datetime.strptime(x, '%m/%d/%Y').year)
    except:        
        pass
    
    try: 
        data['Patient DOB'] = data['Patient DOB'].map(lambda x: x[-4:])
    except:
        pass        
    
    data['Patient DOB'] = data['Patient DOB'].map(lambda x: date.today().year - int(x))
    
    data['Age'] = np.nan
    for i in range(12,0,-1):
        data.loc[data['Patient DOB'] <= i*10, 'Age'] = i           
            
    Feature_object = ['HCPC','Modifier','Dx Code','Patient State','Insurance Name','Claim Type','Clinic ID']
    Feature_float = ['SO Dollars','Age']
    
    for o in Feature_object:
        data[o] = data[o].astype('object')
    for f in Feature_float:
        data[f] = data[f].astype('float64')
        
    final = data[['Patient_SO_ID','HCPC','Modifier','Dx Code','Patient State','Insurance Name','Claim Type',
                  'SO Dollars','Clinic ID','Age','Insurance ID', 'Dr. ID','HCPC_dummy','Modifier_dummy']] 
        
#        
    return final
       
