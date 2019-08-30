import warnings
warnings.filterwarnings('ignore')

import catboost as cat
from .utils import *

X_train, X_val, y_train, y_val, cust_group = load_data(random_state=2018)

cat_features_idx = list(range(96-1, 96 + 60 - 1 )) # 96 - 157 中间缺110, 112
train_pool = cat.Pool(X_train, y_train, cat_features=cat_features_idx)
val_pool = cat.Pool(X_val, y_val, cat_features=cat_features_idx)

params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'objective': 'Logloss', 
    'custom_metric': 'AUC',
    'eval_metric': 'AUC',
    'early_stopping_rounds': 10,
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True
}

best_model = cat.CatBoostClassifier(**params)
best_model.fit(train_pool, eval_set=val_pool)

y_train_pred = best_model.predict_proba(X_train)[:, 1]
y_val_pred = best_model.predict_proba(X_val)[:, 1]


evaluate(y_train, y_train_pred, y_val, y_val_pred, cust_group)
