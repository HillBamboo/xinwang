##  弱到不行的baseline 瞎搞一波 线上0.72835  线下不知道  不会写评测函数....
## 大家尝试将算法改成svm, xgboost, lightGBM 之类的  调调参数 看看效果怎样

# 使用70% 0.7921130703042608
# 使用全部数据 0.8345719704867983
# 使用 svc  0.9956473351180608  划分训练集和验证集后 验证集 0.6032551798935024   overfitting!!!

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_data(data_path='../input/train_xy.csv', test_size=0.3):
    """

    :param data_path: 载入数据的路径
    :param test_size:
    :return:
    """
    data = pd.read_csv(data_path, index_col=0)
    X = data.iloc[:, 2:]
    y = data.loc[:, 'y']
    cust_group = data[['cust_group']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test, cust_group

def save_result(save_fp, pred_prob):
    print('保存预测结果')
    result = {'pred_prob': pred_prob}
    result = pd.DataFrame(result)
    result.to_csv(save_fp)
    print('OK')

def evaluate(y_true, y_preds, cust_group):
    """注意这里的cust_group是所有数据的分组信息， y_true和y_preds只包含少数测试数据"""
    groups = cust_group.loc[y_true.index]['cust_group']
    tmp = pd.DataFrame({'cust_group': groups, 'y_true': y_true, 'y_pred': y_preds})

    group_1 = tmp[tmp['cust_group'] == 'group_1']
    group_2 = tmp[tmp['cust_group'] == 'group_2']
    group_3 = tmp[tmp['cust_group'] == 'group_3']

    auc1 = roc_auc_score(group_1['y_true'], group_1['y_pred'])
    auc2 = roc_auc_score(group_2['y_true'], group_2['y_pred'])
    auc3 = roc_auc_score(group_3['y_true'], group_3['y_pred'])

    auc = 0.3*auc1 + 0.3*auc2 + 0.4*auc3
    print('auc得分: ', auc)


#
# if __name__ == '__main__':
X_train, X_test, y_train, y_test, cust_group = load_data()

print('训练模型:')
clf = LogisticRegression()
# clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

print('预测概率：')
res = clf.predict_proba(X_test)
y_preds = res[:, 1]

evaluate(y_test, y_preds, cust_group)

test_data = pd.read_csv('../input/test_all.csv')
xx = test_data.iloc[:, 2:]
pred_res = clf.predict_proba(xx)
# test_data['cust_id']
# pred_res[:, 1]








