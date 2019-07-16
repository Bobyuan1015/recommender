import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
from scipy.sparse.construct import hstack
from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression  # Logistic 回归模型 包
from sklearn.linear_model import LogisticRegressionCV # 带有正则化参数C的粒度
from sklearn.model_selection import cross_val_score # 交叉验证


def xgboost_lr():
    #导入数据
    data = pd.read_csv('traningset.csv')
    #应变量
    y_all=data['label']

    x_all=data[data.columns[1:-2]]
    print("input:",x_all.columns)
    # x_all=data.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome',
    #              'NumberOfOpenCreditLinesAndLoans',
    #              'NumberRealEstateLoansOrLines', 'NumberOfDependents'],axis=1)
    # 训练/测试数据分割
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2)

    # xgb_model
    xgb_model = XGBClassifier(
         learning_rate =0.2,
         n_estimators=10,
         max_depth=4,
         min_child_weight=1,
         gamma=0.24,
         subsample=0.7,
         colsample_bytree=0.6,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)

    xgb_model.fit(x_train, y_train)
    y_test_predict = xgb_model.predict_proba(x_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, y_test_predict)
    print('xgboost test auc: %.5f' % xgb_test_auc)

    # lr
    lr= LogisticRegression(C=1.0,penalty='l2')#用参数指定网格搜索对模型的正则化参数C的粒度
    lr.fit(x_train, y_train)  # 预测及AUC评测
    y_test_predict = lr.predict_proba(x_test)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_test_predict)
    print('基于原有特征的LR AUC: %.5f' % lr_test_auc)



    # mode= LogisticRegression(C=1.0,penalty='l2')
    # scores = cross_val_score(mode, x_train, y_train, cv = 4,scoring='roc_auc') # 做四次交叉验证
    # y_test_predict = mode.predict_proba(x_test)[:, 1]
    # lr_test_auc = roc_auc_score(y_test, y_test_predict)
    # print('交叉验证4fold 基于原有特征的LR AUC: %.5f' % lr_test_auc)
 # logisticregressioncv 类用参数指定网格搜索对模型的正则化参数C的粒度
    model_cv = LogisticRegressionCV(10)
    model_cv.fit(x_train, y_train)
    y_test_predict = model_cv.predict_proba(x_test)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_test_predict)
    print('基于原有特征的LR AUC: %.5f' % lr_test_auc)


    # xgb编码特征,特征转换,在训练得到34棵树之后，我们需要得到的不是GBDT的预测结果，而是每一条训练数据落在了每棵树的哪个叶子结点上
    x_train_leaves = xgb_model.apply(x_train)
    x_test_leaves = xgb_model.apply(x_test)
    # 合并编码后的训练数据和测试数据
    all_leaves = np.concatenate((x_train_leaves, x_test_leaves), axis=0)
    all_leaves = all_leaves.astype(np.int32)
    # 对所有特征进行ont-hot编码
    one_hot_enc = OneHotEncoder()
    x_trans = one_hot_enc.fit_transform(all_leaves)

    # xgb+lr
    (train_rows, cols) = x_train_leaves.shape
    lr = LogisticRegression()
    lr.fit(x_trans[:train_rows, :], y_train)
    y_xgb_lr1_predict = lr.predict_proba(x_trans[train_rows:, :])[:, 1]
    xgb_lr1_auc = roc_auc_score(y_test, y_xgb_lr1_predict)
    print('基于Xgb编码特征的xgb+lr AUC: %.5f' % xgb_lr1_auc)

    # xgb+lr组合特征
    lr = LogisticRegression()
    # 水平将数组堆叠起来
    X_train_ext = hstack([x_trans[:train_rows, :], x_train])
    X_test_ext = hstack([x_trans[train_rows:, :], x_test])
    lr.fit(X_train_ext, y_train)
    y_xgb_lr2_predict = lr.predict_proba(X_test_ext)[:, 1]
    xgb_lr2_auc = roc_auc_score(y_test, y_xgb_lr2_predict)
    print('基于xgb编码特征和LR组合特征的xgb+lr AUC: %.5f' % xgb_lr2_auc)

xgboost_lr()
