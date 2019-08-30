import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import xgboost as xgb


from .utils import *

X_train, X_val, y_train, y_val, cust_group = load_data_resampled(test_size=0.3)
print("X_train.shape: ", X_train.shape)
print("X_val.shape: ", X_val.shape)

clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=80,
                        silent=True, objective="binary:logistic",
                        booster='gbtree',min_child_weight=3,gamma=0)

# early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_eval, y_eval)]

X_train, y_train = xgb.DMatrix(X_train), xgb.DMatrix(y_train)
X_val, y_val = xgb.DMatrix(X_val), xgb.DMatrix(y_val)

clf.fit(X_train, y_train)
y_train_pred = clf.predict_proba(X_train)[:, 1]
y_val_pred = clf.predict_proba(X_val)[:, 1]
evaluate(y_train, y_train_pred, y_val, y_val_pred, cust_group)