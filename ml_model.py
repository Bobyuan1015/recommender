#-*- coding: utf-8 -*-
'''
Created on 2019-07-13
To do a LR model for recommendation system
@author: yuanfang
'''

from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import StratifiedKFold #交叉验证
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
 #导入数据
data = pd.read_csv('data/ods_sql_traningset.csv')
y_all=data['label']

x_all=data[data.columns[1:-2]]
print("input:",x_all.columns)
# x_all=data.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome',
#              'NumberOfOpenCreditLinesAndLoans',
#              'NumberRealEstateLoansOrLines', 'NumberOfDependents'],axis=1)
# 训练/测试数据分割
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2)
#

lr= LogisticRegression()#用参数指定网格搜索对模型的正则化参数C的粒度
# lr.fit(x_train, y_train)  # 预测及AUC评测
# y_test_predict = lr.predict_proba(x_test)[:, 1]
# lr_test_auc = roc_auc_score(y_test, y_test_predict)
# print('基于原有特征的LR AUC: %.5f' % lr_test_auc)

learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3] #学习率
gamma = [1, 0.1, 0.01, 0.001]

# param_grid = dict(learning_rate = learning_rate,gamma = gamma)#转化为字典格式，网络搜索要求

kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)#将训练/测试数据集划分10个互斥子集，

param_grid={'C':[0.1,0.3,0.5,0.7,0.9,1.0,1.2,1.4,1.6,2.0], 'penalty':['l1','l2']}
grid_search = GridSearchCV(lr,param_grid,scoring = 'f1',n_jobs = -1,cv = kflod)
#scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
grid_result = grid_search.fit(x_all, y_all) #运行网格搜索
print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
