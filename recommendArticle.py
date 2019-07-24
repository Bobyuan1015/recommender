#-*- coding: utf-8 -*-
'''
Created on 2019-07-13
Main process for recommendation system
@author: yuanfang
'''
from cf_model import Cf_model
from dataProcess import DataProcess


if __name__=='__main__':
    dataProcess=DataProcess()
    path=dataProcess.featureBuilding()
    usercf = Cf_model()
    usercf.generate_dataset(path)
    usercf.calc_user_sim()
    dataProcess.toHabse(usercf)


    # 以下为用户id为100的用户推荐的资讯
    # a = usercf.recommend("100")
    # cost time: 2.384185791015625e-05
    # [(186, 0.5, 1), (128, 0.5, 2)...
    # usercf.evaluate()
