import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

application_train = pd.read_csv('../input/application_train.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
credit_card_balance = credit_card_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

# count cards
ccb_prev_count = credit_card_balance.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
# INSTALMENTS
credit_card_balance['PAY_MONTH'] = credit_card_balance.CNT_INSTALMENT_MATURE_CUM.map(lambda x: 1 if x > 0 else 0)
ccb_temp = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['PAY_MONTH'].sum().reset_index()
ccb_avg_inst_card = ccb_temp.groupby('SK_ID_CURR')['PAY_MONTH'].mean()
ccb_total_inst_card = ccb_temp.groupby('SK_ID_CURR')['PAY_MONTH'].sum()
# limit
ccb_temp = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['AMT_CREDIT_LIMIT_ACTUAL'].mean().reset_index()
ccb_avg_limit_card = ccb_temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
ccb_max_limit_card = credit_card_balance.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].max()
ccb_total_limit_card = ccb_temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum()
# avg drawing amount
ccb_temp = credit_card_balance.loc[credit_card_balance.CNT_DRAWINGS_CURRENT>0].groupby(['SK_ID_CURR','MONTHS_BALANCE'])['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT'].sum().reset_index()
ccb_temp['avg_drawing_amount'] = (ccb_temp.AMT_DRAWINGS_CURRENT / ccb_temp.CNT_DRAWINGS_CURRENT).fillna(0)
ccb_avg_drawing_amount = ccb_temp.groupby('SK_ID_CURR')['avg_drawing_amount'].mean().fillna(0)
# count Refused
ccb_count_rej = credit_card_balance.groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg(lambda x: np.sum(x=='Refused'))
last_month_credit = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
last_month_credit = last_month_credit.merge(credit_card_balance, on=['SK_ID_CURR','SK_ID_PREV', 'MONTHS_BALANCE'])
# current credit card situation
ccb_cur_total_receivable = last_month_credit.groupby('SK_ID_CURR')['AMT_TOTAL_RECEIVABLE'].sum()
ccb_cur_total_limit = last_month_credit.loc[last_month_credit.NAME_CONTRACT_STATUS == 'Active'].groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum() # fillna: 0
ccb_cur_total_payment = last_month_credit.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum()
ccb_cur_total_balance = last_month_credit.groupby('SK_ID_CURR')['AMT_BALANCE'].sum()
# drawing in 1y
ccb_temp = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-12]
ccb_drawing_amount_1y = ccb_temp.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum()
ccb_drawing_times_1y = ccb_temp.groupby('SK_ID_CURR')['CNT_DRAWINGS_CURRENT'].sum()
# drawing in 6m
ccb_temp = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-6]
ccb_drawing_amount_6m = ccb_temp.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum()
ccb_drawing_times_6m = ccb_temp.groupby('SK_ID_CURR')['CNT_DRAWINGS_CURRENT'].sum()
# DPD
ccb_temp = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
ccb_max_dpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
ccb_total_dpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
ccb_max_largedpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
ccb_total_largedpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

ccb_feature = pd.DataFrame({'ccb_prev_count':ccb_prev_count, 'ccb_avg_inst_card':ccb_avg_inst_card, 'ccb_avg_limit_card':ccb_avg_limit_card, 'ccb_total_inst_card':ccb_total_inst_card, 'ccb_count_rej': ccb_count_rej,
                        'ccb_avg_limit_card':ccb_avg_limit_card, 'ccb_max_limit_card':ccb_max_limit_card, 'ccb_total_limit_card':ccb_total_limit_card, 'ccb_avg_drawing_amount':ccb_avg_drawing_amount,
                        'ccb_cur_total_receivable':ccb_cur_total_receivable, 'ccb_cur_total_limit':ccb_cur_total_limit, 'ccb_cur_total_payment':ccb_cur_total_payment, 'ccb_cur_total_balance':ccb_cur_total_balance,
                        'ccb_drawing_amount_1y':ccb_drawing_amount_1y, 'ccb_drawing_times_1y':ccb_drawing_times_1y, 'ccb_drawing_amount_6m':ccb_drawing_amount_6m, 'ccb_drawing_times_6m':ccb_drawing_times_6m,
                        'ccb_max_dpd_days':ccb_max_dpd_days, 'ccb_total_dpd_days':ccb_total_dpd_days, 'ccb_max_largedpd_days':ccb_max_largedpd_days, 'ccb_total_largedpd_days':ccb_total_largedpd_days}).reset_index()
ccb_feature.to_csv('ccb_feature.csv', index=False)