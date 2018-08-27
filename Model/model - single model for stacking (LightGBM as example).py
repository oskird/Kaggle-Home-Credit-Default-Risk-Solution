# import modules
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
trail = None

# baseline model
folds = StratifiedKFold(n_splits=5, random_state=6)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_pred.shape[0])

start = time.time()
valid_score = 0
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    trn_x, trn_y = X.iloc[trn_idx], y[trn_idx]
    val_x, val_y = X.iloc[val_idx], y[val_idx]
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y, free_raw_data=True, categorical_feature=new_cat)
    valid_data = lgb.Dataset(data=val_x, label=val_y, free_raw_data=True, categorical_feature=new_cat)
    
    param = {'application':'xentropy','num_iterations':15000, 'learning_rate':0.01, 'num_leaves':36, 'feature_fraction':0.1, 'bagging_fraction':0.9,
             'lambda_l1':3, 'lambda_l2':1, 'min_split_gain':0.01, 'early_stopping_round':100, 'max_depth':7, 'metric':'auc'}
    
    lgb_es_model = lgb.train(param, train_data, valid_sets=[train_data, valid_data], verbose_eval=200, categorical_feature=new_cat) 
    
    oof_preds[val_idx] = lgb_es_model.predict(val_x, num_iteration=lgb_es_model.best_iteration)
    sub_preds += lgb_es_model.predict(X_pred, num_iteration=lgb_es_model.best_iteration) / folds.n_splits
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    valid_score += roc_auc_score(val_y, oof_preds[val_idx])

print('valid score:', str(round(valid_score/folds.n_splits,3)))

end = time.time()
print(str((end-start)/60), 'mins')

# oof train and test prediction
application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
output = pd.DataFrame({'SK_ID_CURR': application_train.SK_ID_CURR,'TARGET': oof_preds})
output.to_csv('valid_pred.csv', index=False)

application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
output = pd.DataFrame({'SK_ID_CURR': application_test.SK_ID_CURR,'TARGET': sub_preds})
output.to_csv('final_pred.csv', index=False)

# feature importance
lgb.plot_importance(lgb_es_model, height=0.5, max_num_features=30, ignore_zero = False, figsize = (12,6), importance_type ='gain')