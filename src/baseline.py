# 使用70% 0.7921130703042608
# 使用全部数据 0.8345719704867983
# 使用 svc  0.9956473351180608  划分训练集和验证集后 验证集 0.6032551798935024   overfitting!!!

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from .utils import *
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cust_group = load_data(random_state=None)

    # 综下，一组好的参数： n_estimators=70, max_depth=21, min_samples_split=230, max_features='sqrt'
    # print('======================= RF调参 =======================')
    # params1 = {'n_estimators': range(10, 101, 10)}
    # clf = RandomForestClassifier(max_features='sqrt', n_jobs=-1)
    # grid1 = GridSearchCV(estimator=clf, param_grid=params1, cv=5, scoring='roc_auc')
    # grid1.fit(X_train, y_train)
    # print(grid1.grid_scores_)
    # print(grid1.best_params_)
    # print(grid1.best_score_)
    # 结果：
    # [mean: 0.67233, std: 0.01794, params: {'n_estimators': 10}, mean: 0.73365, std: 0.02115, params: {
    #     'n_estimators': 20}, mean: 0.73598, std: 0.02068, params: {
    #     'n_estimators': 30}, mean: 0.75495, std: 0.02853, params: {
    #     'n_estimators': 40}, mean: 0.76233, std: 0.03893, params: {
    #     'n_estimators': 50}, mean: 0.75321, std: 0.02761, params: {
    #     'n_estimators': 60}, mean: 0.77370, std: 0.03284, params: {
    #     'n_estimators': 70}, mean: 0.76488, std: 0.02418, params: {
    #     'n_estimators': 80}, mean: 0.76703, std: 0.02254, params: {
    #     'n_estimators': 90}, mean: 0.77037, std: 0.02144, params: {'n_estimators': 100}]
    # {'n_estimators': 70}
    # 0.7736997596388921

    # params2 = {'max_depth': range(3, 11, 2), 'min_samples_split': range(50, 201, 20)}
    # clf = RandomForestClassifier(n_estimators=70, max_features='sqrt', n_jobs=-1)
    # grid2 = GridSearchCV(estimator=clf, param_grid=params2, cv=5, scoring='roc_auc')
    # grid2.fit(X_train, y_train)
    # print(grid2.grid_scores_)
    # print(grid2.best_params_)
    # print(grid2.best_score_)
    # 结果
    # {'max_depth': 9, 'min_samples_split': 190}
    # 0.8028731126473071

    # params3 = {'max_depth': range(9, 31, 2), 'min_samples_split': range(190, 301, 20)}
    # clf = RandomForestClassifier(n_estimators=70, max_features='sqrt', n_jobs=-1)
    # grid3 = GridSearchCV(estimator=clf, param_grid=params3, cv=5, scoring='roc_auc')
    # grid3.fit(X_train, y_train)
    # print(grid3.grid_scores_)
    # print(grid3.best_params_)
    # print(grid3.best_score_)
    # 结果
    # {'max_depth': 21, 'min_samples_split': 230}
    # 0.8148149184875022

    # params3 = {'max_features': ['auto', 'log2', 'sqrt']}
    # clf = RandomForestClassifier(n_estimators=70, max_depth=21, min_samples_split=230, n_jobs=-1)
    # grid3 = GridSearchCV(estimator=clf, param_grid=params3, cv=5, scoring='roc_auc')
    # grid3.fit(X_train, y_train)
    # print(grid3.grid_scores_)
    # print(grid3.best_params_)
    # print(grid3.best_score_)
    # 结果：
    # {'max_features': 'sqrt'}
    # 0.8074729451077706

    # params3 = {'max_features': [5, 10, 13, 15, 20, 25, 30]}
    # clf = RandomForestClassifier(n_estimators=70, max_depth=21, min_samples_split=230, n_jobs=-1)
    # grid3 = GridSearchCV(estimator=clf, param_grid=params3, cv=5, scoring='roc_auc')
    # grid3.fit(X_train, y_train)
    # print(grid3.best_params_)
    # print(grid3.best_score_)
    # 结果：
    # {'max_features': 13}
    # 0.805689820705211

    # clf = RandomForestClassifier(n_estimators=70, max_depth=21, min_samples_split=230, max_features='sqrt', n_jobs=-1)
    # clf.fit(X_train, y_train)
    # predict_auc(clf, X_train, y_train, X_test, y_test, cust_group)
    # 结果：
    # 训练auc: 0.909479 - ----- 测试auc: 0.757845

    print('======================= LR调参 =======================')
    clf = LogisticRegression(penalty='l1', C=1)
    clf.fit(X_train, y_train)
    predict_auc(clf, X_train, y_train, X_test, y_test, cust_group)
    # 结果：
    # {'C': 0.01}
    # 0.810225963898087

#     print('======================= 集成方法 =======================')
    # # pca = PCA(n_components=n)
#     X_train_pca = X_train
#     X_test_pca = X_test
#     clf1 = LogisticRegression(C=0.1)
#     clf2 = RandomForestClassifier(n_estimators=70, max_depth=21, min_samples_split=230, max_features='sqrt', n_jobs=-1)
#     # clf3 = GaussianNB()
#     # clf4 = svm.LinearSVC()
#     eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft', weights=[1,1])
#     # params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200], }
#     # grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
#     # grid = grid.fit(X_train_pca, y_train)

#     for each_clf, label in zip([clf1, clf2, eclf], ['lr', 'rf', 'ensemble']):
#         each_clf.fit(X_train_pca, y_train)
#         y_train_pred = each_clf.predict_proba(X_train_pca)[:, 1]  # 取出类别为1的预测概率值
#         y_test_pred = each_clf.predict_proba(X_test_pca)[:, 1]  # 取出类别为1的预测概率值

#         auc_train = evaluate(y_train, y_train_pred, cust_group)
#         auc_test = evaluate(y_test, y_test_pred, cust_group)
#         print('%s 训练auc: %.6f ------ 测试auc: %.6f' % (label, auc_train, auc_test))
    # lr
    # 训练auc: 0.810824 - ----- 测试auc: 0.759416
    # rf
    # 训练auc: 0.906122 - ----- 测试auc: 0.770817
    # ensemble
    # 训练auc: 0.883639 - ----- 测试auc: 0.782707

    # 训练auc: 0.808695 - ----- 测试auc: 0.783613
    # print('======================= 训练单个LR模型 =======================')
    #
    # print('训练模型:')
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    #
    # print('预测概率：')
    # y_train_pred = clf.predict_proba(X_train)[:, 1] # 取出类别为1的预测概率值
    # y_test_pred = clf.predict_proba(X_test)[:, 1] # 取出类别为1的预测概率值
    #
    # auc_train = evaluate(y_train, y_train_pred, cust_group)
    # auc_test = evaluate(y_test, y_test_pred, cust_group)
    # print('训练auc: %.6f ------ 测试auc: %.6f' %(auc_train, auc_test))

    # 下面的效果会差些
    # group1 训练auc: 0.783991 - ----- 测试auc: 0.709323
    # group1 训练auc: 0.872480 - ----- 测试auc: 0.682655
    # group3 训练auc: 0.919739 - ----- 测试auc: 0.769604
    # 综合 训练auc: 0.864837 - ----- 测试auc: 0.725435
    # print('======================= 训练三个LR模型 =======================')
    # data = pd.read_csv('../input/train_xy.csv', index_col=0)
    # X = data.iloc[:, 2:]
    # y = data.loc[:, 'y']
    #
    # X1 = data[data['cust_group'] == 'group_1'].iloc[:, 2:]
    # y1 = data[data['cust_group'] == 'group_1'].loc[:, 'y']
    # X2 = data[data['cust_group'] == 'group_2'].iloc[:, 2:]
    # y2 = data[data['cust_group'] == 'group_2'].loc[:, 'y']
    # X3 = data[data['cust_group'] == 'group_3'].iloc[:, 2:]
    # y3 = data[data['cust_group'] == 'group_3'].loc[:, 'y']
    #
    # X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
    # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
    # X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3)
    #
    # clf1 = LogisticRegression()
    # clf2 = LogisticRegression()
    # clf3 = LogisticRegression()
    #
    # print('训练模型...')
    # clf1.fit(X1_train, y1_train)
    # clf2.fit(X2_train, y2_train)
    # clf3.fit(X3_train, y3_train)
    #
    # print('预测概率：')
    # y1_train_pred = clf1.predict_proba(X1_train)[:, 1] # 取出类别为1的预测概率值
    # y1_test_pred = clf1.predict_proba(X1_test)[:, 1] # 取出类别为1的预测概率值
    # auc1_train = roc_auc_score(y1_train, y1_train_pred)
    # auc1_test = roc_auc_score(y1_test, y1_test_pred)
    # print('group1 训练auc: %.6f ------ 测试auc: %.6f' %(auc1_train, auc1_test))
    #
    #
    # y2_train_pred = clf2.predict_proba(X2_train)[:, 1] # 取出类别为1的预测概率值
    # y2_test_pred = clf2.predict_proba(X2_test)[:, 1] # 取出类别为1的预测概率值
    # auc2_train = roc_auc_score(y2_train, y2_train_pred)
    # auc2_test = roc_auc_score(y2_test, y2_test_pred)
    # print('group1 训练auc: %.6f ------ 测试auc: %.6f' %(auc2_train, auc2_test))
    #
    #
    # y3_train_pred = clf3.predict_proba(X3_train)[:, 1] # 取出类别为1的预测概率值
    # y3_test_pred = clf3.predict_proba(X3_test)[:, 1] # 取出类别为1的预测概率值
    # auc3_train = roc_auc_score(y3_train, y3_train_pred)
    # auc3_test = roc_auc_score(y3_test, y3_test_pred)
    # print('group3 训练auc: %.6f ------ 测试auc: %.6f' %(auc3_train, auc3_test))
    #
    # auc_train = 0.3*auc1_train + 0.3*auc2_train + 0.4*auc3_train
    # auc_test = 0.3*auc1_test + 0.3*auc2_test + 0.4*auc3_test
    # print('\n\n综合 训练auc: %.6f ------ 测试auc: %.6f' %(auc_train, auc_test))