import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc

bureau_balance = pd.read_csv('../input/bureau_balance.csv')

# bb feature
bureau_balance['STATUS_mod'] = bureau_balance.STATUS.map({'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, 'X':np.nan, 'C':0}).map(lambda x: 0 if x=='C' else x).interpolate(method = 'linear')
bureau_balance['write_off'] = bureau_balance.STATUS.map(lambda x: 1 if x=='5' else 0)
bureau_balance['adj_score'] = (bureau_balance.MONTHS_BALANCE-bureau_balance.MONTHS_BALANCE.min()+1)*bureau_balance.STATUS_mod

bb_month_count = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count()
bb_dpd_sum = bureau_balance.groupby('SK_ID_BUREAU')['STATUS_mod'].sum()
bb_write_off = bureau_balance.groupby('SK_ID_BUREAU')['write_off'].sum()
bb_dpd_sum_2_year = bureau_balance.loc[bureau_balance.MONTHS_BALANCE>=-24].groupby('SK_ID_BUREAU')['STATUS_mod'].sum()
bb_write_off_2_year = bureau_balance.loc[bureau_balance.MONTHS_BALANCE>=-24].groupby('SK_ID_BUREAU')['write_off'].sum()
bb_adj_score = bureau_balance.groupby('SK_ID_BUREAU')['adj_score'].sum()

bb_feature = pd.DataFrame({'bb_month_count':bb_month_count, 'bb_dpd_sum':bb_dpd_sum, 'bb_write_off':bb_write_off, 'bb_dpd_sum_2_year':bb_dpd_sum_2_year, 
                          'bb_write_off_2_year':bb_write_off_2_year, 'bb_adj_score': bb_adj_score}).reset_index().fillna(0)
del bb_month_count, bb_dpd_sum, bb_write_off, bb_dpd_sum_2_year, bb_write_off_2_year, bb_adj_score, bureau_balance
gc.collect()

# bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau = bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'])

bureau['ADJ_DAYS'] = (bureau.DAYS_CREDIT - bureau.DAYS_CREDIT.min()) / (bureau.DAYS_CREDIT.max() - bureau.DAYS_CREDIT.min()) + 0.5 # more recent, more effecitve
# application count
bur_ncount = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()
bur_act_count = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
bur_bad_count = bureau.loc[bureau.CREDIT_ACTIVE=='Bad debt'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
bur_sold_count = bureau.loc[bureau.CREDIT_ACTIVE=='Sold out'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
# application date
bur_recent_application = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max()
bur_eariliest_application = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
bur_max_enddate = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].max()
# application itervel
bureau['application_interval'] = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].diff(-1)
missing_iter = iter(bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max())
bureau.application_interval = bureau.application_interval.map(lambda x: -next(missing_iter) if np.isnan(x) else -x)
bur_avg_intervel = bureau.groupby('SK_ID_CURR')['application_interval'].mean()
bur_sd_intervel = bureau.groupby('SK_ID_CURR')['application_interval'].agg('std').fillna(0)
# overdue days
bur_max_overdue_days = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max()
bur_active_total_overdue_days = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].sum()
bur_active_max_overdue_days = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max()
bureau['DAYS_CREDIT_ISPAST'] = bureau.DAYS_CREDIT_ENDDATE.map(lambda x: 1 if x < 0 else 0)
bur_avg_remaining_days = bureau.loc[bureau.DAYS_CREDIT_ENDDATE>0].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()
# overdue amount
bureau['ADJ_AMT_CREDIT_MAX_OVERDUE'] = bureau.ADJ_DAYS * bureau.AMT_CREDIT_MAX_OVERDUE
bur_total_max_overdue_adj = bureau.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT_MAX_OVERDUE'].sum() # use adj days
bur_avg_max_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].mean()
bur_overall_max_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].max()
# adj prelong
bureau['ADJ_CNT_CREDIT_PROLONG'] = bureau.ADJ_DAYS * bureau.CNT_CREDIT_PROLONG
bur_avg_prelonged = bureau.groupby('SK_ID_CURR')['ADJ_CNT_CREDIT_PROLONG'].mean().fillna(0) # use adj days
bur_max_prelonged = bureau.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].max().fillna(0)
bur_total_prelonged_adj = bureau.groupby('SK_ID_CURR')['ADJ_CNT_CREDIT_PROLONG'].sum().fillna(0)
# historical amount
bureau['ADJ_AMT_CREDIT_SUM_DEBT'] = bureau.ADJ_DAYS * bureau.AMT_CREDIT_SUM
bur_total_amount_adj = bureau.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT_SUM_DEBT'].sum()
bur_avg_amount = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean()
# current amount
bur_active_total_amount = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum() # fillna 0
bur_active_avg_amount = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean() # fillna 0
bur_active_total_debt = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum() # fillna 0
bur_active_avg_debt = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].mean() # fillna 0
bur_active_total_limit  = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].sum() # fillna 0
bur_active_avg_limit  = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].mean() # fillna 0
bur_active_total_overdue = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum() # fillna 0
bur_active_avg_overdue = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].mean() # fillna 0
bur_active_ratio_debt_credit = (bur_active_total_debt / bur_active_total_amount.map(lambda x: x+0.1)) # fillna 0
bur_active_ratio_overdue_debt = (bur_active_total_overdue / bur_active_total_debt.map(lambda x: x+0.1)) # fillna 0
# credit update
bur_avg_update = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].mean()
bur_recent_update = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].max()
# annuity
bur_avg_annuity = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean() # can't fillna 0
bur_total_annuity = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].sum() # can't fillna 0
bur_active_total_annuity = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_ANNUITY'].sum()
bureau['term'] = bureau.AMT_CREDIT_SUM / bureau.AMT_ANNUITY
bur_avg_term = bureau.loc[bureau.term < float('inf')].groupby('SK_ID_CURR')['term'].mean() # can't fillna 0

bureau_num_feature = pd.DataFrame({'bur_ncount':bur_ncount, 'bur_act_count':bur_act_count, 'bur_bad_count':bur_bad_count, 'bur_sold_count':bur_sold_count,
            'bur_recent_application':bur_recent_application,'bur_eariliest_application':bur_eariliest_application, 'bur_max_enddate':bur_max_enddate,
            'bur_avg_intervel':bur_avg_intervel, 'bur_sd_intervel':bur_sd_intervel,
            'bur_max_overdue_days':bur_max_overdue_days, 'bur_active_total_overdue_days':bur_active_total_overdue_days, 'bur_active_max_overdue_days':bur_active_max_overdue_days,
            'bur_avg_remaining_days':bur_avg_remaining_days, 'bur_total_max_overdue_adj':bur_total_max_overdue_adj, 'bur_avg_max_overdue':bur_avg_max_overdue, 'bur_overall_max_overdue':bur_overall_max_overdue, 
            'bur_avg_prelonged':bur_avg_prelonged, 'bur_max_prelonged':bur_max_prelonged, 'bur_total_prelonged_adj':bur_total_prelonged_adj,
            'bur_total_amount_adj':bur_total_amount_adj, 'bur_avg_amount':bur_avg_amount,
            'bur_active_total_amount':bur_active_total_amount, 'bur_active_avg_amount':bur_active_avg_amount, 'bur_active_total_debt':bur_active_total_debt, 'bur_active_avg_debt':bur_active_avg_debt, 
            'bur_active_total_limit':bur_active_total_limit, 'bur_active_avg_limit':bur_active_avg_limit, 'bur_active_total_overdue':bur_active_total_overdue, 'bur_active_avg_overdue':bur_active_avg_overdue,
            'bur_active_ratio_debt_credit':bur_active_ratio_debt_credit, 'bur_active_ratio_overdue_debt':bur_active_ratio_overdue_debt,
            'bur_avg_update':bur_avg_update, 'bur_recent_update':bur_recent_update,
            'bur_avg_annuity':bur_avg_annuity, 'bur_total_annuity':bur_total_annuity, 'bur_active_total_annuity':bur_active_total_annuity, 'bur_avg_term':bur_avg_term}).reset_index()
fill0_list = ['bur_act_count', 'bur_bad_count', 'bur_sold_count', 'bur_active_total_overdue_days', 'bur_active_max_overdue_days', 'bur_active_total_amount', 'bur_active_avg_amount',
             'bur_active_total_debt', 'bur_active_avg_debt', 'bur_active_total_limit', 'bur_active_avg_limit', 'bur_active_total_overdue', 'bur_active_avg_overdue', 
              'bur_active_ratio_debt_credit', 'bur_active_ratio_overdue_debt', 'bur_active_total_annuity']
bureau_num_feature[fill0_list] = bureau_num_feature[fill0_list] .fillna(0)

bureau_cat = pd.get_dummies(bureau[['SK_ID_CURR','CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']], prefix='bur')
bureau_cat_feature = bureau_cat.groupby('SK_ID_CURR').mean().reset_index()
del bureau_cat
gc.collect()

bureau_bb = bureau[['SK_ID_CURR','SK_ID_BUREAU']].merge(bb_feature, on='SK_ID_BUREAU', how='left')

bb_avg_month = bureau_bb.groupby('SK_ID_CURR')['bb_month_count'].mean()
bb_total_overdue_month = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum'].sum()
bb_total_writeoff = bureau_bb.groupby('SK_ID_CURR')['bb_write_off'].sum()
bb_max_overdue_month = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum'].max()
bb_max_writeoff = bureau_bb.groupby('SK_ID_CURR')['bb_write_off'].max()

bb_total_overdue_month_2year = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum_2_year'].sum()
bb_max_overdue_month_2year= bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum_2_year'].max()
bb_total_writeoff_2year = bureau_bb.groupby('SK_ID_CURR')['bb_write_off_2_year'].sum()
bb_max_writeoff_2year = bureau_bb.groupby('SK_ID_CURR')['bb_write_off_2_year'].max()

bb_max_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].max()
bb_total_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].sum()
bb_avg_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].mean()

bureau_bb_feature = pd.DataFrame({'bb_avg_month':bb_avg_month, 
            'bb_total_overdue_month':bb_total_overdue_month, 'bb_total_writeoff':bb_total_writeoff, 'bur_sold_count':bb_max_overdue_month,'bb_max_writeoff':bb_max_writeoff,
            'bb_total_overdue_month_2year':bb_total_overdue_month_2year, 'bb_total_writeoff_2year':bb_total_writeoff_2year, 
            'bb_max_overdue_month_2year':bb_max_overdue_month_2year,'bb_max_writeoff_2year':bb_max_writeoff_2year, 
            'bb_max_score':bb_max_score, 'bb_total_score':bb_total_score, 'bb_avg_score':bb_avg_score}).reset_index()
			
bureau_feature = bureau_num_feature.merge(bureau_cat_feature, on='SK_ID_CURR').merge(bureau_bb_feature, on='SK_ID_CURR', how='left')
print(bureau_feature.shape)
bureau_feature.to_csv('bureau_feature.csv', index=False)