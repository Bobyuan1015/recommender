# coding: utf-8 -*-
# To do a feature engineering for recommendation system
# Author:yuanfang
# Date:2019-07-13
import math
import pandas as pd
import time
import random

class UserCf:

    def __init__(self):
        self.file_path = 'postive_data.csv'
        self._init_frame()

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)
        self.frame['Rating']= random.randint(1,5)

    @staticmethod
    def _cosine_sim(target_news, news):
        '''
             Cosine Similarity - Definition - Ochiai Coefficient
        '''
        union_len = len(set(target_news) & set(news))#交集
        if union_len == 0: return 0.0
        product = len(target_news) * len(news)
        cosine = union_len / math.sqrt(product)
        return cosine

    def _get_top_n_users(self, target_user_id, top_n):
        '''
        calculate similarity between all users and return Top N similar users.
        '''
        target_news = self.frame[self.frame['userid'] == target_user_id]['newsid']
        other_users_id = [i for i in set(self.frame['userid']) if i != target_user_id]
        other_news = [self.frame[self.frame['userid'] == i]['newsid'] for i in other_users_id]

        sim_list = [self._cosine_sim(target_news, news) for news in other_news]
        sim_list = sorted(zip(other_users_id, sim_list), key=lambda x: x[1], reverse=True)
        return sim_list[:top_n]

    def _get_candidates_items(self, target_user_id):
        """
        Find all news in source data and target_user did not meet before.
        """
        target_user_news = set(self.frame[self.frame['userid'] == target_user_id]['newsid'])
        other_user_news = set(self.frame[self.frame['userid'] != target_user_id]['newsid'])
        candidates_news = list(target_user_news ^ other_user_news)
        return candidates_news

    def _get_top_n_items(self, top_n_users, candidates_news, top_n):
        """
        calculate interest of candidates news and return top n news.
        e.g. interest = sum(sim * normalize_rating)
        """
        top_n_user_data = [self.frame[self.frame['userid'] == k] for k, _ in top_n_users]
        interest_list = []
        for news_id in candidates_news:
            tmp = []
            for user_data in top_n_user_data:
                if news_id in user_data['newsid'].values:
                    tmp.append(user_data[user_data['newsid'] == news_id]['Rating'].values[0]/5)
                else:
                    tmp.append(0)
            interest = sum([top_n_users[i][1] * tmp[i] for i in range(len(top_n_users))])
            interest_list.append((news_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]

    def calculate(self, target_user_id=82, top_n=3):
        """
        user-cf for news recommendation.
        """
        # most similar top n users
        start = time.time()
        top_n_users = self._get_top_n_users(target_user_id, top_n)
        print('top_n_users  Cost time: %f' % (time.time() - start))
        # candidates news for recommendation
        candidates_news = self._get_candidates_items(target_user_id)
        print('candidates_news  Cost time: %f' % (time.time() - start))
        # most interest top n news
        top_n_news = self._get_top_n_items(top_n_users, candidates_news, top_n)
        print('top_n_news  Cost time: %f' % (time.time() - start))
        return top_n_news


if __name__=='__main__':
    UserCf().calculate(82,2)
