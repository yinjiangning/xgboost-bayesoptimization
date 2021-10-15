import pandas as pd
import gc
import warnings

from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import contextlib

'''
def read_dataset(fname):
    data = pd.read_csv(fname, encoding="utf-8")
    # drop掉无用数据
    data.drop(['obs_time', 'patient_id', 'dataset_name'], axis=1, inplace=True)
    return data


train = read_dataset('190624_data.csv')
y = train['is_vte_bool'].values
X = train.drop(['is_vte_bool'], axis=1).values
y = y + 0  # 将布尔值替换成01

# 数据归一化
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X = min_max_scaler.fit_transform(X)
cols = train.columns

# 把数据的80%用来训练模型,20%做模型测试和评估,此处用到训练集-验证集二划分
p = 0.8  # 设置训练数据比例,
X_train = X[:int(len(X) * p), :]  # 前80%为训练集
X_test = X[int(len(X) * p):, :]  # 后20%为测试集
y_train = y[:int(len(y) * p)]  # 前80%为训练集
y_test = y[int(len(y) * p):]  # 后20%为测试集

'''
data = pd.read_csv('D:/s-sn/ASCII1/evidence1.csv',encoding='gbk')


#testdata=minmax_scale(testdata)
# 划分数据
x = data.drop('status',axis=1)
y = data.status
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =2018, shuffle = True)

# 数据归一化
#X_train = minmax_scale(X_train)
#X_test =  minmax_scale(X_test)
# Comment out any parameter you don't want to test
def XGB_CV(
        learning_rate,
        max_depth,
        gamma,
        min_child_weight,
        max_delta_step,
        subsample,
        colsample_bytree
):
    global AUCbest
    global ITERbest

    # Define all XGboost parameters
    paramt = {
        'learning_rate': learning_rate,
        'booster': 'gbtree',
        'max_depth': int(max_depth),
        'gamma': gamma,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'nthread': 4,
        'silent': True,
        'eval_metric': 'auc',
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'min_child_weight': min_child_weight,
        'max_delta_step': int(max_delta_step),
        'seed': 1001
    }

    folds = 5
    cv_score = 0

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file)
    log_file.flush()

    xgbc = xgb.cv(
        paramt,
        dtrain,
        num_boost_round=20000,
        stratified=True,
        nfold=folds,
        #                    verbose_eval = 10,
        early_stopping_rounds=100,
        metrics='auc',
        show_stdv=True
    )

    val_score = xgbc['test-auc-mean'].iloc[-1]
    train_score = xgbc['train-auc-mean'].iloc[-1]
    print(
        ' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % (
            len(xgbc), train_score, val_score, (train_score - val_score), (train_score * 2 - 1), (val_score * 2 - 1)))
    if (val_score > AUCbest):
        AUCbest = val_score
        ITERbest = len(xgbc)

    return (val_score * 2) - 1


log_file = open('Porto-AUC-5fold-XGB-run-01-v1-full.log', 'a')
AUCbest = -1.
ITERbest = 0
dtrain = xgb.DMatrix(X_train, label=y_train)

XGB_BO = BayesianOptimization(XGB_CV, {
    'learning_rate': (0.01, 1),
    'max_depth': (2, 12),
    'gamma': (0.001, 10.0),
    'min_child_weight': (0, 20),
    'max_delta_step': (0, 10),
    'subsample': (0.4, 1.0),
    'colsample_bytree': (0.4, 1.0)
})

XGB_BO.explore({
    'learning_rate': [0.01, 0.03, 0.01, 0.03, 0.1, 0.3, 0.1, 0.3],
    'max_depth': [3, 8, 3, 8, 8, 3, 8, 3],
    'gamma': [0.5, 8, 0.2, 9, 0.5, 8, 0.2, 9],
    'min_child_weight': [0.2, 0.2, 0.2, 0.2, 12, 12, 12, 12],
    'max_delta_step': [1, 2, 2, 1, 2, 1, 1, 2],
    'subsample': [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
})

print('-' * 130)
print('-' * 130, file=log_file)
log_file.flush()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=2, n_iter=5, acq='ei', xi=0.0)

print('-' * 130)
print('Final Results')
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])
print('-' * 130, file=log_file)
print('Final Result:', file=log_file)
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'], file=log_file)
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'], file=log_file)
log_file.flush()
log_file.close()

history_df = pd.DataFrame(XGB_BO.res['all']['params'])
history_df2 = pd.DataFrame(XGB_BO.res['all']['values'])
history_df = pd.concat((history_df, history_df2), axis=1)
history_df.rename(columns={0: 'gini'}, inplace=True)
history_df['AUC'] = (history_df['gini'] + 1) / 2
history_df.to_csv('D:/Porto-AUC-5fold-XGB-run-01-v1-grid.csv')
