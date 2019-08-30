import pandas as pd
import numpy as np


from .utils import  *

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV

# load or create your dataset
print('Load data...')
X_train, X_test, y_train, y_test, cust_group = load_data(random_state=None)

# TODO 上采样  + 重复特征
# X_train_1_30 = X_train.iloc[:, :100]
# X_test_1_30 = X_test.iloc[:, :100]

# X_train = pd.concat([X_train, X_train, X_train, X_train], axis=1)
# X_test = pd.concat([X_test, X_test, X_test, X_test], axis=1)


if True:
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['x_' + str(i) for i in range(96, 157+1)])
    lgb_eval = lgb.Dataset(X_test, y_test, categorical_feature=['x_' + str(i) for i in range(96, 157+1)], reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'max_bin': 127,
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'objective': 'binary',
        'metric': ['binary_logloss'],
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'verbose': 0,
        'num_iteration': 1000,
        'lambda_l2': 0.5,
        'lambda_l1': 0.9,
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10
                    )

#     print('Save model...')
#     # save model to file
#     gbm.save_model('../model/lgb_tuned.txt')

    print('Start predicting...')
    # predict
    y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    auc_train = evaluate(y_train, y_train_pred, cust_group)
    auc_test = evaluate(y_test, y_test_pred, cust_group)
    print('auc: {train}/{test}'.format(train=auc_train, test=auc_test))

    # print('Save predict result...')
    # test_data = pd.read_csv('../input/test_all.csv')
    # test_data.replace(-99, np.nan, inplace=True)
    # X = test_data.iloc[:, 2:]
    # # y = test_data.loc[:, 'y']
    # pred_prob = gbm.predict(X)
    # result = pd.DataFrame({'cust_id': test_data['cust_id'], 'pred_prob': pred_prob})
    # result.to_csv('../output/lgb_21_21.csv', index=None)
    # print('OK')

    # train/test  不处理 分类变量  没有处理缺失值  train test  有一些gap   lgb_default.csv
    # (0.8815845375511302, 0.82150974561878765)
    # train/test  处理 分类变量 没有处理缺失值  train test gap 小了一些  lgb_categorical.csv  TODO 提交这个！！！
    # (0.83593374592744241, 0.81249912267596203)
    # train/test  处理 分类变量 处理缺失值  overfitting   lgb_categorical_nan.csv
    # 0.9451635459576995/0.8203973377652068
    # train/test  max_depth = 17   lgb_maxdepth_17.csv
    # 0.9565878333782296/0.8199572923243731
    # train/test  max_depth = 9   lgb_maxdepth_9.csv
    # 0.9472050637088709/0.8260295621879845
    # train/test  max_depth = 21, num_leave = 21  lgb_bagging_fraction.csv
    # 0.8593299279209903 / 0.8218309032347775    TODO: 这个效果很不错！！！

    # 使用2000条测试数据
    # auc: 0.9445612202121074 / 0.8536013759004015

# print('========================= lgb调参 =========================')
# estimator = lgb.LGBMClassifier(num_leaves=31, objective='binary', metric='auc', verbose=0)
#
# param_grid = {
#     'max_depth': range(15, 51, 2)
# }
#
# gbm = GridSearchCV(estimator, param_grid)
#
# gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', categorical_feature=range(96, 157+1),early_stopping_rounds=5)
# print(gbm.grid_scores_)
# print(gbm.best_params_)
# print(gbm.best_score_)
# 结果：
# [mean: 0.95314, std: 0.00001, params: {'max_depth': 3}, mean: 0.95305, std: 0.00035, params: {'max_depth': 5}, mean: 0.95286, std: 0.00039, params: {'max_depth': 7}, mean: 0.95352, std: 0.00014, params: {'max_depth': 9}, mean: 0.95352, std: 0.00014, params: {'max_depth': 11}, mean: 0.95305, std: 0.00035, params: {'max_depth': 13}, mean: 0.95295, std: 0.00067, params: {'max_depth': 15}, mean: 0.95343, std: 0.00071, params: {'max_depth': 17}, mean: 0.95343, std: 0.00071, params: {'max_depth': 19}, mean: 0.95343, std: 0.00071, params: {'max_depth': 21}]
# {'max_depth': 9}
# 0.953523809524
