import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score

def bayes_parameter_opt_lgb(X, y, init_round=10, opt_round=20, n_folds=3, random_seed=6):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = new_cat, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations':500, 'learning_rate':0.05, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 40),
                                            'feature_fraction': (0.1, 0.2),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (6, 9.99),
                                            'lambda_l1': (0, 10),
                                            'lambda_l2': (0, 1),
                                            'min_split_gain': (0.008, 0.015),
                                            'min_child_weight': (1, 100)}, random_state=random_seed)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # output optimization process
    lgbBO.points_to_csv("bayes_opt_result.csv")
    # return best parameters
    return lgbBO.res['max']['max_params']
opt_params = bayes_parameter_opt_lgb(X, y, init_round=10, opt_round=40, n_folds=3, random_seed=3)
print(opt_params)