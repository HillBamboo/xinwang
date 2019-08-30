import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import GridSearchCV
import pickle

try:
    from src.utils import *
except:
    try:
        from .utils import *
    except:
        try:
            from utils import *
        except:
            print('不好意思，我已经尽力-_-')
            print('import error for importing utils package')

# print("采样前:")
# X_train, X_val, y_train, y_val, cust_group = load_data(test_size=0.3)
# print("X_train.shape: ", X_train.shape)
# print("X_val.shape: ", X_val.shape)
# # print("y_train: ", str(set(y_train)))
#
# clf1 = LogisticRegression(penalty='l1')
# clf1.fit(X_train, y_train)
# y_train_pred, y_val_pred = clf1.predict_proba(X_train)[:, 1], clf1.predict_proba(X_val)[:, 1]
# evaluate(y_train, y_train_pred, y_val, y_val_pred, cust_group)
#


print("采样后:")
X_resampled_train, X_resampled_val, y_resampled_train, y_resampled_val, cust_group = load_data_resampled(test_size=0.3)
print("X_resampled_train.shape: ", X_resampled_train.shape)
print("X_resampled_val.shape: ", X_resampled_val.shape)

pkl_fp = b'../model/lr.pkl'
clf2 = None # load_model_if_exist(pkl_fp)
if not clf2:
    clf2 = LogisticRegression(penalty='l1', C=0.01)
    clf2.fit(X_resampled_train, y_resampled_train)
    save_model(clf2, pkl_fp)

print('Predicting...')
y_train_pred, y_val_pred = clf2.predict_proba(X_resampled_train)[:, 1], clf2.predict_proba(X_resampled_val)[:, 1]
evaluate(y_resampled_train, y_train_pred, y_resampled_val, y_val_pred, cust_group)

cust_id, x_test = load_data('../input/test_all.csv', flag='test')
preds = clf2.predict_proba(x_test)[:, 1]
make_submission('../output/lr_smote.csv', preds, np.ravel(cust_id))

