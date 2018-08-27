import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

application_train = pd.read_csv('../input/application_train.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
POS_CASH_balance = POS_CASH_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

pcb_prev_count = POS_CASH_balance.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
pcb_avg_month = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].count()/pcb_prev_count
pcb_recent_active = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()

# times of INSTALMENT change
pcb_temp_inst_change_time = POS_CASH_balance[['SK_ID_PREV', 'CNT_INSTALMENT']].groupby('SK_ID_PREV')['CNT_INSTALMENT'].nunique().map(lambda x: x - 1).reset_index().rename(columns={'CNT_INSTALMENT':'pcb_prev_inst_change_time'})
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].count().reset_index()
pcb_temp = pcb_temp.merge(pcb_temp_inst_change_time, on='SK_ID_PREV')
pcb_inst_change_time = pcb_temp.groupby('SK_ID_CURR')['pcb_prev_inst_change_time'].sum()

# avg INSTALMENT
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT'].mean().reset_index()
pcb_avg_inst = pcb_temp.groupby(['SK_ID_CURR'])['CNT_INSTALMENT'].mean()

# active INSTALMENT
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
pcb_temp = pcb_temp.merge(POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS']], on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
pcb_temp['active_1'] = pcb_temp.MONTHS_BALANCE.map(lambda x: 1 if x>=-4 else 0)
pcb_temp['active_2'] = pcb_temp.NAME_CONTRACT_STATUS.map(lambda x: 1 if x=='Active' else 0)
pcb_temp['active'] = pcb_temp.active_1 * pcb_temp.active_2
pcb_active_inst = pcb_temp.groupby('SK_ID_CURR')['active'].count()

# DPD
pcb_temp = POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_temp = POS_CASH_balance.loc[POS_CASH_balance.MONTHS_BALANCE>=-12, ['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_temp = POS_CASH_balance.loc[POS_CASH_balance.MONTHS_BALANCE>=-24, ['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_num = pd.DataFrame({'pcb_prev_count':pcb_prev_count, 'pcb_avg_month':pcb_avg_month, 'pcb_recent_active':pcb_recent_active, 'pcb_inst_change_time':pcb_inst_change_time,
                       'pcb_avg_inst':pcb_avg_inst, 'pcb_active_inst':pcb_active_inst, 
                       'pcb_max_dpd_days':pcb_max_dpd_days, 'pcb_total_dpd_days':pcb_total_dpd_days, 'pcb_max_largedpd_days':pcb_max_largedpd_days, 'pcb_total_largedpd_days':pcb_total_largedpd_days,
                       'pcb_max_dpd_days_1y':pcb_max_dpd_days_1y, 'pcb_total_dpd_days_1y':pcb_total_dpd_days_1y, 'pcb_max_largedpd_days_1y':pcb_max_largedpd_days_1y, 
                       'pcb_total_largedpd_days_1y':pcb_total_largedpd_days_1y, 'pcb_max_dpd_days_2y':pcb_max_dpd_days_2y, 'pcb_total_dpd_days_2y':pcb_total_dpd_days_2y, 
                       'pcb_max_largedpd_days_2y':pcb_max_largedpd_days_2y, 'pcb_total_largedpd_days_2y':pcb_total_largedpd_days_2y,}).reset_index()
					   
# INSTALMENT: ratio of end status of each type
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
pcb_temp = pcb_temp.merge(POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS']], on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
pcb_temp_dummy = pd.get_dummies(pcb_temp, prefix='pcb_end_as')
pcb_end_as_dummy = pcb_temp_dummy.loc[pcb_temp_dummy.pcb_end_as_Active!=1].drop(['SK_ID_PREV','MONTHS_BALANCE', 'pcb_end_as_Active'], axis=1).groupby('SK_ID_CURR').sum().reset_index()
del pcb_temp_dummy
gc.collect()

pcb_feature = pcb_num.merge(pcb_end_as_dummy, on='SK_ID_CURR', how='left')

pcb_feature['pcb_no_dpd'] = pcb_feature.pcb_total_dpd_days.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd'] = pcb_feature.pcb_total_largedpd_days.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_dpd_1y'] = pcb_feature.pcb_total_dpd_days_1y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd_1y'] = pcb_feature.pcb_total_largedpd_days_1y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_dpd_2y'] = pcb_feature.pcb_total_dpd_days_2y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd_2y'] = pcb_feature.pcb_total_largedpd_days_2y.map(lambda x: 0 if x== 0 else 1)
pcb_feature.to_csv('pcb_feature.csv', index=False)