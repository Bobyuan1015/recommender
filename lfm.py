# coding: utf-8 -*-
import random
import pickle
import pandas as pd
import numpy as np
import os
import math
from math import exp
import sys


class Corpus:

    items_dict_path = 'lfm_items.dict'

    @classmethod
    def pre_process(cls):
        file_path = 'data/ods_sql_lfm_postive_data.csv'
        cls.frame = pd.read_csv(file_path)

        cls.user_ids = set(cls.frame['useridReindex'].values)
        cls.item_ids = set(cls.frame['newsidReindex'].values)
        cls.items_dict = {user_id: cls._get_pos_neg_item(user_id) for user_id in list(cls.user_ids)}
        cls.save()

    @classmethod
    def _get_pos_neg_item(cls, user_id):
        """
        Define the pos and neg item for user.
        pos_item mean items that user have rating, and neg_item can be items
        that user never see before.
        Simple down sample method to solve unbalance sample.
        """
        print('Process: {}'.format(user_id))
        pos_item_ids = set(cls.frame[cls.frame['useridReindex'] == user_id]['newsidReindex'])
        neg_item_ids = cls.item_ids ^ pos_item_ids
        # neg_item_ids = [(item_id, len(self.frame[self.frame['newsidReindex'] == item_id]['useridReindex'])) for item_id in neg_item_ids]
        # neg_item_ids = sorted(neg_item_ids, key=lambda x: x[1], reverse=True)
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

    @classmethod
    def save(cls):
        f = open(cls.items_dict_path, 'wb')
        pickle.dump(cls.items_dict, f)
        f.close()

    @classmethod
    def load(cls):
        f = open(cls.items_dict_path, 'rb')
        items_dict = pickle.load(f)
        f.close()
        return items_dict


class LFM:

    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lam = 0.01
        self.trainset = {}
        self.testset = {}
        # 训练集用的相似用户数
        self.n_sim_user = 20
        # 推荐资讯数量
        self.n_rec_article = 10
        self.user_sim_mat = {}
        self.article_popular = {}
        self.article_count = 0
        print ('相似用户数目为 = %d' % self.n_sim_user, file=sys.stderr)
        print ('推荐资讯数目为 = %d' %
               self.n_rec_article, file=sys.stderr)
        self._init_model()


    def _init_model(self):
        """
        Get corpus and initialize model params.
        """
        file_path = 'lfm_postive_data.csv'
        self.frame = pd.read_csv(file_path)
        self.user_ids = set(self.frame['useridReindex'].values)
        self.item_ids = set(self.frame['newsidReindex'].values)
        self.items_dict = Corpus.load()

        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def _recommend(self, user_id, item_id):
        """
        Calculate interest between user_id and item_id.
        p is the look-up-table for user's interest of each class.
        q means the probability of each item being classified as each class.
        """
        p = np.mat(self.p.ix[user_id].values)
        q = np.mat(self.q.ix[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + exp(-r))
        return logit

# self._loss(user_id, item_id, item_dict[item_id], step)
    def _loss(self, user_id, item_id, y, step):
        """
        Loss Function define as MSE, the code write here not that formula you think.
        """
        e = y - self._recommend(user_id, item_id)
        print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.
              format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        """
        Use SGD as optimizer, with L2 p, q square regular.
        e.g: E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
             derivation(E, p) = -matrix_q*(y - predict), derivation(E, q) = -matrix_p*(y - predict),
             derivation（l2_square，p) = lam * p, derivation（l2_square, q) = lam * q
             delta_p = lr * (derivation(E, p) + derivation（l2_square，p))
             delta_q = lr * (derivation(E, q) + derivation（l2_square, q))
        """
        gradient_p = -e * self.q.ix[item_id].values
        l2_p = self.lam * self.p.ix[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.ix[user_id].values
        l2_q = self.lam * self.q.ix[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        """
        Train model.
        """
        for step in range(0, self.iter_count):
            for user_id, item_dict in self.items_dict.items():
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    def recommend(self, user_id, top_n=10):
        """
        Calculate all item user have not meet before and return the top n interest items.
        """
        self.load()
        user_item_ids = set(self.frame[self.frame['useridReindex'] == user_id]['newsidReindex'])
        other_item_ids = self.item_ids ^ user_item_ids
        interest_list = [self._recommend(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def save(self):
        """
        Save model params.
        """
        f = open('lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        """
        Load model params.
        """
        f = open('lfm.model', 'rb')
        self.p, self.q = pickle.load(f)
        f.close()

 # 划分训练集和测试集 pivot用来定义训练集和测试集的比例

# 计算 准确略，召回率，覆盖率，流行度
    def evaluate(self, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        for i in self.frame.index:
       
            user=self.frame.useridReindex[i]
            article=self.frame.newsidReindex[i]
            # 随机数字 如果小于0.7 则数据划分为训练集
            if random.random() < pivot:
                # 设置训练集字典，key为user,value 为字典 且初始为空
                self.trainset.setdefault(user, {})
                # 以下省略格式如下，集同一个用户id 会产生一个字典，且值为他评分过的所有资讯
                #{'1': {'914': 3, '3408': 4, '150': 5, '1': 5}, '2': {'1357': 5}}
                self.trainset[user][article] =0# int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][article] = 0#int(rating)
                testset_len += 1
        # 输出切分训练集成功
        print ('划分数据为训练集和测试集成功！', file=sys.stderr)
        # 输出训练集比例
        print ('训练集数目 = %s' % trainset_len, file=sys.stderr)
        # 输出测试集比例
        print ('测试集数目 = %s' % testset_len, file=sys.stderr)





        # ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)
        f = open('cf.model', 'rb')
        self.user_sim_mat,self.article_popular,self.article_count = pickle.load(f)
        f.close()

        N = self.n_rec_article
        #  varables for precision and recall
        #记录推荐正确的资讯数

        hit = 0
        #记录推荐资讯的总数
        rec_count = 0
        #记录测试数据中总数
        test_count = 0
        # varables for coverage
        all_rec_articles = set()
        # varables for popularity
        popular_sum = 0

        for ii, user1 in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % ii, file=sys.stderr)
            test_articles = self.testset.get(user1, {})
            rec_articles = self.recommend(user1)
            for article1, _ in rec_articles:
                if article1 in test_articles:
                    hit += 1
                all_rec_articles.add(article1)
                # popular_sum += math.log(1 + self.article_popular[article1])
            rec_count += N
            test_count += len(test_articles)
        # 计算准确度
        precision = hit / (1.0 * rec_count)
        # 计算召回率
        recall = hit / (1.0 * test_count)
        # 计算覆盖率
        coverage = len(all_rec_articles) / (1.0 * self.article_count)
        #计算流行度
        # popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\t' %
               (precision, recall, coverage), file=sys.stderr)

if __name__=='__main__':
    if not os.path.exists('lfm_items.dict'):
        Corpus.pre_process()
    if not os.path.exists('lfm.model'):
        LFM().train()
    # movies=LFM().recommend(82,2)
    # for movie in movies:
    #     print(movie)
    LFM().evaluate()
