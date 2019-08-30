import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from utils import *


def drop_replace(data, drop_cols, inplace=True):
	data.drop(drop_cols, axis=1, inplace=inplace)
	data.replace(-99, -1, inplace=inplace)

print('Loading data......', end='')
train_data = pd.read_csv('../input/train_xy.csv', index_col=0)
drop_replace(train_data, ['x_110', 'x_112'])
cust_group_train, X, y = train_data.cust_group, train_data.iloc[:, 2:], train_data.y
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1000, random_state=20181001)
print('\033[0;37;42mDone!\033[0m')

test_data = pd.read_csv('../input/test_all.csv')
drop_replace(test_data, ['x_110', 'x_112'])
X_test, cust_group_test = test_data.iloc[:, 2:], test_data.cust_group

#xgboost，lr模型按3:2比例融合
clf= XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=80, silent=True,
	objective="binary:logistic", booster='gbtree',min_child_weight=3,subsample=0.8,gamma=0)
clf2= LogisticRegression(C=0.1, penalty='l2', tol=1e-4)
clf4= RandomForestClassifier(n_estimators=400,oob_score=True)
eclf = VotingClassifier(estimators=[('xgb', clf), ('lr', clf2)], voting='soft',weights=[1.5,1])

print('Training Model......', end='')
eclf.fit(X_train, y_train)
print('\033[0;37;42mDone!\033[0m')

print('Predicting......', end='')
y_train_preds = eclf.predict_proba(X_train)[:, 1]
y_val_preds = eclf.predict_proba(X_val)[:, 1]
print('\033[0;37;42mDone!\033[0m')

print('Saving file......', end='')
for i in range(2, 10):
    tmp1, tmp2 = np.around(y_train_preds, i), np.around(y_val_preds, i)
    evaluate(y_train, tmp1, y_val, tmp2, train_data)
    # print(f'{i} precision for train auc score: {train_auc}') # 0.87575078357674 for 无舍入
print('\033[0;37;42mDone!\033[0m')
print('\033[0;37;44mNice!!!\033[0m')


