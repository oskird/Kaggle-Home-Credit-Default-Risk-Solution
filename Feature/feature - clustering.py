import numpy as np
import pandas as pd
import gc

bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
med = bureau.AMT_CREDIT_SUM.median()
bureau.AMT_CREDIT_SUM = bureau.AMT_CREDIT_SUM.fillna(med)
med = bureau.AMT_CREDIT_SUM_DEBT.median()
bureau.AMT_CREDIT_SUM_DEBT = bureau.AMT_CREDIT_SUM_DEBT.fillna(med)
bureau['OVERDUE_DEBT_RATIO'] = bureau.AMT_CREDIT_SUM_OVERDUE/(bureau.AMT_CREDIT_SUM_DEBT+1)
bureau['DEBT_TOTAL_RATIO'] = bureau.AMT_CREDIT_SUM_DEBT/(bureau.AMT_CREDIT_SUM+1)
bureau_balance['INT_STATUS'] = bureau_balance.STATUS.replace('X', 0.1).replace('C', 0).astype('int64')
bur_max_bad_level = bureau_balance.groupby('SK_ID_BUREAU')['INT_STATUS'].max().reset_index()
cluter_bur = bureau[['SK_ID_BUREAU', 'CREDIT_DAY_OVERDUE', 'OVERDUE_DEBT_RATIO', 'DEBT_TOTAL_RATIO', 'CNT_CREDIT_PROLONG']].merge(bur_max_bad_level, on='SK_ID_BUREAU', how='left').fillna(0)

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
X = cluter_bur.drop(['SK_ID_BUREAU'], axis=1)
X = Normalizer().fit_transform(X)
gmm = GaussianMixture(n_components=2, verbose=5, max_iter=100, init_params='kmeans')
gmm.fit(X)
group_prob = gmm.predict_proba(X)
group_prob = np.round(group_prob, decimals=2)
bur_cluster = pd.concat([bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], pd.DataFrame({'cluster':group_prob[:, 0]})], axis=1)
bur_cluster = bur_cluster.groupby('SK_ID_CURR')['cluster'].mean().reset_index()
bur_cluster.to_csv('bur_cluster.csv', index=False)
