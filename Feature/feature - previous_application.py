import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

previous_application = pd.read_csv("../input/previous_application.csv")
previous_application = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

# cat
cat_col = []
for i in range(len(previous_application.columns)):
    if previous_application.iloc[:, i].dtype == 'object':
        cat_col.append(i)
cat_pa = previous_application.iloc[:, cat_col]
cat_pa = pd.concat([previous_application[['HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_INSURED_ON_APPROVAL']],cat_pa], axis=1).fillna('XNA')
cat_pa.HOUR_APPR_PROCESS_START = cat_pa.HOUR_APPR_PROCESS_START.astype('object')
cat_pa.NFLAG_LAST_APPL_IN_DAY = cat_pa.NFLAG_LAST_APPL_IN_DAY.astype('object')
cat_pa.NFLAG_INSURED_ON_APPROVAL = cat_pa.NFLAG_INSURED_ON_APPROVAL.astype('object')
# selected version
cat_NAME_CONTRACT_TYPE = pd.get_dummies(cat_pa.NAME_CONTRACT_TYPE, prefix='pa_NAME_CONTRACT_TYPE').drop('pa_NAME_CONTRACT_TYPE_XNA', axis=1)
cat_NAME_CASH_LOAN_PURPOSE = pd.get_dummies(cat_pa.NAME_CASH_LOAN_PURPOSE, prefix='pa_NAME_CASH_LOAN_PURPOSE').drop('pa_NAME_CASH_LOAN_PURPOSE_XNA', axis=1)
cat_NAME_CONTRACT_STATUS = pd.get_dummies(cat_pa.NAME_CONTRACT_STATUS, prefix='pa_NAME_CONTRACT_STATUS')
cat_NAME_PAYMENT_TYPE = pd.get_dummies(cat_pa.NAME_PAYMENT_TYPE, prefix='pa_NAME_PAYMENT_TYPE').drop('pa_NAME_PAYMENT_TYPE_XNA', axis=1)
cat_CODE_REJECT_REASON = pd.get_dummies(cat_pa.CODE_REJECT_REASON, prefix='pa_CODE_REJECT_REASON').drop('pa_CODE_REJECT_REASON_XNA', axis=1)
cat_NAME_CLIENT_TYPE = pd.get_dummies(cat_pa.NAME_CLIENT_TYPE, prefix='pa_NAME_CLIENT_TYPE').drop('pa_NAME_CLIENT_TYPE_XNA', axis=1)
cat_NAME_PORTFOLIO = pd.get_dummies(cat_pa.NAME_PORTFOLIO, prefix='pa_NAME_PORTFOLIO').drop('pa_NAME_PORTFOLIO_XNA', axis=1)
cat_NAME_PRODUCT_TYPE = pd.get_dummies(cat_pa.NAME_PRODUCT_TYPE, prefix='pa_NAME_PRODUCT_TYPE').drop('pa_NAME_PRODUCT_TYPE_XNA', axis=1)
cat_NAME_YIELD_GROUP = pd.get_dummies(cat_pa.NAME_YIELD_GROUP, prefix='pa_NAME_YIELD_GROUP').drop('pa_NAME_YIELD_GROUP_XNA', axis=1)
cat_PRODUCT_COMBINATION = pd.get_dummies(cat_pa.PRODUCT_COMBINATION, prefix='pa_PRODUCT_COMBINATION').drop('pa_PRODUCT_COMBINATION_XNA', axis=1)
cat_NFLAG_INSURED_ON_APPROVAL = pd.get_dummies(cat_pa.NFLAG_INSURED_ON_APPROVAL, prefix='pa_NFLAG_INSURED_ON_APPROVAL').drop('pa_NFLAG_INSURED_ON_APPROVAL_XNA', axis=1)
cat_pa_dummy = pd.concat([previous_application[['SK_ID_PREV', 'SK_ID_CURR']], cat_NAME_CONTRACT_TYPE, cat_NAME_CASH_LOAN_PURPOSE, cat_NAME_CONTRACT_STATUS, cat_NAME_PAYMENT_TYPE, cat_CODE_REJECT_REASON, 
                          cat_NAME_CLIENT_TYPE, cat_NAME_PORTFOLIO, cat_NAME_PRODUCT_TYPE, cat_NAME_YIELD_GROUP, cat_PRODUCT_COMBINATION, cat_NFLAG_INSURED_ON_APPROVAL], axis=1)
del cat_NAME_CONTRACT_TYPE, cat_NAME_CASH_LOAN_PURPOSE, cat_NAME_CONTRACT_STATUS, cat_NAME_PAYMENT_TYPE, cat_CODE_REJECT_REASON, cat_NAME_CLIENT_TYPE, 
cat_NAME_PORTFOLIO, cat_NAME_PRODUCT_TYPE, cat_NAME_YIELD_GROUP, cat_PRODUCT_COMBINATION, cat_NFLAG_INSURED_ON_APPROVAL, cat_pa
gc.collect()
pa_cat = cat_pa_dummy.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR').mean().reset_index()

# num
num_col = []
for i in range(len(previous_application.columns)):
    if (previous_application.iloc[:, i].dtype == 'int64') or (previous_application.iloc[:, i].dtype == 'float64'):
        num_col.append(i)
num_pa = previous_application.iloc[:, num_col].drop(['HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_INSURED_ON_APPROVAL'], axis=1)
for i in range(-5, 0):
    num_pa.iloc[:, i] = num_pa.iloc[:, i].map(lambda x: np.nan if x==365243.0 else x)
# create new variables
num_pa['app_vs_actual_less'] = (num_pa.AMT_APPLICATION < num_pa.AMT_CREDIT).map(lambda x: 1 if x==True else 0)
num_pa['app_vs_actual_more'] = (num_pa.AMT_APPLICATION > num_pa.AMT_CREDIT).map(lambda x: 1 if x==True else 0)
a = num_pa.DAYS_DECISION - num_pa.DAYS_DECISION.min() + 1
num_pa['ADJ_SCORE'] = (a-a.min()) / (a.max()-a.min()) + 0.5
num_pa['ADJ_AMT_ANNUITY'] = num_pa['ADJ_SCORE']*num_pa['AMT_ANNUITY']
num_pa['ADJ_AMT_APPLICATION'] = num_pa['ADJ_SCORE']*num_pa['AMT_APPLICATION']
num_pa['ADJ_AMT_CREDIT'] = num_pa['ADJ_SCORE']*num_pa['AMT_CREDIT']
num_pa['FREQ'] = (num_pa['DAYS_LAST_DUE']-num_pa['DAYS_FIRST_DUE'])/num_pa['CNT_PAYMENT']
pa_prev_count = num_pa.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
pa_avg_annuity = num_pa.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean()
pa_avg_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].mean()
pa_avg_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].mean()
pa_max_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].max()
pa_max_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].max()
pa_total_annuity = num_pa.groupby('SK_ID_CURR')['AMT_ANNUITY'].sum()
pa_total_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].sum()
pa_total_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].sum()
# compare application and actual credit
pa_not_full_credit_times = num_pa.groupby('SK_ID_CURR')['app_vs_actual_less'].sum()
pa_not_full_credit_rate = num_pa.groupby('SK_ID_CURR')['app_vs_actual_less'].mean()
pa_get_more_credit_times = num_pa.groupby('SK_ID_CURR')['app_vs_actual_more'].sum()
pa_get_more_credit_rate = num_pa.groupby('SK_ID_CURR')['app_vs_actual_more'].mean()
# adjusted: more recent, more important
pa_total_annuity_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_ANNUITY'].sum()
pa_total_application_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_APPLICATION'].sum()
pa_total_actual_credit_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT'].sum()
# down payment
pa_avg_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].mean()
pa_max_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].max()
pa_total_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].sum()
# goods price
pa_avg_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].mean()
pa_max_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].max()
pa_total_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].sum()
# down payment rate
pa_avg_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean()
pa_max_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].max()
pa_total_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].sum()
# selling area
pa_avg_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].mean()
pa_max_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].max()
pa_total_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].sum()
# term
pa_avg_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].mean()
pa_max_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].max()
pa_total_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].sum()
pa_most_frequent_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].agg(pd.Series.mode).map(lambda x: np.mean(x))
# recent decision
pa_recent_decision_day = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].max()
pa_earliest_decision_day = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].min()
pa_usage_length = pa_recent_decision_day - pa_earliest_decision_day
# application interval
num_pa['application_interval'] = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].diff(-1)
missing_iter = iter(num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].max())
num_pa.application_interval = num_pa.application_interval.map(lambda x: -next(missing_iter) if np.isnan(x) else -x)
pa_avg_intervel = num_pa.groupby('SK_ID_CURR')['application_interval'].mean()
pa_sd_intervel = num_pa.groupby('SK_ID_CURR')['application_interval'].agg('std').fillna(0)
# days
pa_first_due_day = num_pa.groupby('SK_ID_CURR')['DAYS_FIRST_DUE'].min()
pa_last_due_day = num_pa.groupby('SK_ID_CURR')['DAYS_LAST_DUE'].max()
pa_last_termination_day = num_pa.groupby('SK_ID_CURR')['DAYS_TERMINATION'].max()
pa_lastdue_termination_range = pa_last_termination_day - pa_last_due_day
pa_avg_freq = num_pa.groupby('SK_ID_CURR')['FREQ'].mean()
pa_min_freq = num_pa.groupby('SK_ID_CURR')['FREQ'].min()
pa_num = pd.DataFrame({'pa_prev_count':pa_prev_count, 'pa_avg_annuity':pa_avg_annuity, 'pa_avg_application':pa_avg_application, 'pa_avg_actual_credit':pa_avg_actual_credit, 
'pa_max_application':pa_max_application, 'pa_max_actual_credit':pa_max_actual_credit, 'pa_total_annuity':pa_total_annuity, 'pa_total_application':pa_total_application, 'pa_total_actual_credit':pa_total_actual_credit,
'pa_not_full_credit_times':pa_not_full_credit_times, 'pa_not_full_credit_rate':pa_not_full_credit_rate, 'pa_get_more_credit_times':pa_get_more_credit_times, 'pa_get_more_credit_rate':pa_get_more_credit_rate,
'pa_total_annuity_adj':pa_total_annuity_adj, 'pa_total_application_adj':pa_total_application_adj, 'pa_total_actual_credit_adj':pa_total_actual_credit_adj,
'pa_avg_down_payment':pa_avg_down_payment, 'pa_max_down_payment':pa_max_down_payment, 'pa_total_down_payment':pa_total_down_payment,
'pa_avg_goods_price':pa_avg_goods_price, 'pa_max_goods_price':pa_max_goods_price, 'pa_total_goods_price':pa_total_goods_price,
'pa_avg_down_payment_rate':pa_avg_down_payment_rate, 'pa_max_down_payment_rate':pa_max_down_payment_rate, 'pa_total_down_payment_rate':pa_total_down_payment_rate,
'pa_avg_selling_area':pa_avg_selling_area, 'pa_max_selling_area':pa_max_selling_area, 'pa_total_selling_area':pa_total_selling_area,
'pa_avg_term':pa_avg_term, 'pa_max_term':pa_max_term, 'pa_total_term':pa_total_term, 'pa_most_frequent_term':pa_most_frequent_term,
'pa_recent_decision_day':pa_recent_decision_day, 'pa_earliest_decision_day':pa_earliest_decision_day, 'pa_usage_length': pa_usage_length,
'pa_avg_intervel':pa_avg_intervel, 'pa_sd_intervel':pa_sd_intervel, 'pa_first_due_day':pa_first_due_day, 'pa_last_due_day':pa_last_due_day, 
'pa_last_termination_day':pa_last_termination_day, 'pa_lastdue_termination_range':pa_lastdue_termination_range, 'pa_avg_freq':pa_avg_freq, 'pa_min_freq':pa_min_freq}).reset_index()
pa_feature = pa_num.merge(pa_cat, on='SK_ID_CURR')
del pa_num, pa_cat
gc.collect()
pa_feature.to_csv('pa_feature.csv', index=False)