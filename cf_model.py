#-*- coding: utf-8 -*-
'''
Created on 2019-07-13
To do a cf model for recommendation system
@author: yuanfang
'''
import sys
import random
import math
import time
import os
from operator import itemgetter
import pickle
import pandas as pd
from collections import defaultdict

random.seed(0)


class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    # 构造函数，用来初始化
    def __init__(self):
        # 定义 训练集 测试集 为字典类型
        self.trainset = {}
        self.testset = {}
        # 训练集用的相似用户数
        self.n_sim_user = 1000
        # 推荐电影数量
        self.n_rec_article = 100

        self.user_sim_mat = {}
        self.users=[]
        self.items=[]
        self.article_popular = {}
        self.article_count = 0
        # sys.stderr 是用来重定向标准错误信息的
        print ('相似用户数目为 = %d' % self.n_sim_user, file=sys.stderr)
        print ('推荐数目为 = %d' %
               self.n_rec_article, file=sys.stderr)


    # 加载文件
    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        # 以只读的方式打开传入的文件
        fp = open(filename, 'r')
        # enumerate()为枚举，i为行号从0开始，line为值
        for i, line in enumerate(fp):
            # yield 迭代去下一个值，类似next()
                # line.strip()用于去除字符串头尾指定的字符。
            yield line.strip('\r\n')
            # 计数
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        # 打印加载文件成功
        print ('load %s succ' % filename, file=sys.stderr)

    # 划分训练集和测试集 pivot用来定义训练集和测试集的比例
    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        df=pd.read_csv(filename)
        self.users=list(set(df.userid.to_list()))
        self.items=list(set(df.newsid.to_list()))


        for i in df.index:
            # 根据 分隔符 :: 来切分每行数据
            # userid, newsid, nickname, name, sex, head_url, region, autograph, integral, is_big_v, is_first_login, creator_id, news_title, news_subtitle, updated_year, updated_month, updated_day, updated_timestamp, label, useridReindex, newsidReindex=line.split(',')
            # user, article, rating, _ = line.split('::')
            user=df.userid[i]
            article=df.newsid[i]
            self.trainset.setdefault(user, {})
            # 以下省略格式如下，集同一个用户id 会产生一个字典，且值为他评分过的所有电影
            #{'1': {'914': 3, '3408': 4, '150': 5, '1': 5}, '2': {'1357': 5}}
            self.trainset[user][article] =0# int(rating)
            trainset_len += 1
            # 随机数字 如果小于0.7 则数据划分为训练集
#             if random.random() < pivot:
#                 # 设置训练集字典，key为user,value 为字典 且初始为空
#                 self.trainset.setdefault(user, {})
#                 # 以下省略格式如下，集同一个用户id 会产生一个字典，且值为他评分过的所有电影
#                 #{'1': {'914': 3, '3408': 4, '150': 5, '1': 5}, '2': {'1357': 5}}
#                 self.trainset[user][article] =0# int(rating)
#                 trainset_len += 1
#             else:
#                 self.testset.setdefault(user, {})
#                 self.testset[user][article] = 0#int(rating)
#                 testset_len += 1
        print("index",df.index)
        print("len(users)",len(self.users),' ',self.users[-10:-1])
        print("len(items)",len(self.items),' ',self.items[-10:-1])
        print("len(trainset)",len(self.trainset),' ',list(self.trainset)[-10:-1])
        print('')
        # 输出切分训练集成功
        print ('划分数据为训练集和测试集成功！', file=sys.stderr)
        # 输出训练集比例
        print ('训练集数目 = %s' % trainset_len, file=sys.stderr)
        # 输出测试集比例
        print ('测试集数目 = %s' % testset_len, file=sys.stderr)
    # 建立物品-用户 倒排表
    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=articleID, value=list of userIDs who have seen this article
        print ('构建物品-用户倒排表中，请等待......', file=sys.stderr)
        article2users = dict()

        # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
        for user, articles in self.trainset.items():
            for article in articles:
                # inverse table for item-users
                if article not in article2users:
                    # 根据电影id 构造set() 函数创建一个无序不重复元素集
                    article2users[article] = set()
                # 集合中值为用户id
                # 数值形如
                # {'914': {'1','6','10'}, '3408': {'1'} ......}
                article2users[article].add(user)
                # 记录电影的流行度
                if article not in self.article_popular:
                    self.article_popular[article] = 0
                self.article_popular[article] += 1
        print ('构建物品-用户倒排表成功', file=sys.stderr)

        # save the total article number, which will be used in evaluation
        self.article_count = len(article2users)
        print ('总共被操作过的电影数目为 = %d' % self.article_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat

        print ('building user co-rated articles matrix...', file=sys.stderr)
        # 令系数矩阵 C[u][v]表示N(u)∩N（v) ，假如用户u和用户v同时属于K个物品对应的用户列表，就有C[u][v]=K
        for article, users in article2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated articles matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        # 循环遍历usersim_mat 根据余弦相似度公式计算出用户兴趣相似度
        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                # 以下是公式计算过程
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                #计数 并没有什么卵用
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)
        f = open('cf.model', 'wb')

        pickle.dump((self.user_sim_mat,self.article_popular,self.article_count), f)
        f.close()
    # 根据用户给予推荐结果
    def recommend(self, user):
        '''定义给定K个相似用户和推荐N个电影'''

        f = open('cf.model', 'rb')
        self.user_sim_mat,self.article_popular,self.article_count = pickle.load(f)
        f.close()

        start = time.time()
        K = self.n_sim_user
        N = self.n_rec_article
        # 定义一个字典来存储为用户推荐的电影
        rank = dict()
        watched_articles = self.trainset[user]
        # sorted() 函数对所有可迭代的对象进行排序操作。 key 指定比较的对象 ，reverse=True 降序
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for article in self.trainset[similar_user]:
                # 判断 如果这个电影 该用户已经看过 则跳出循环
                if article in watched_articles:
                    continue
                # 记录用户对推荐的电影的兴趣度
                rank.setdefault(article, 0)
                rank[article] += similarity_factor
        # return the N best articles
        print('cost time:',time.time()-start)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    # 计算 准确略，召回率，覆盖率，流行度
    def evaluate(self):

        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        # f = open('cf.model', 'rb')
        # self.user_sim_mat,self.article_popular,self.article_count = pickle.load(f)
        # f.close()

        N = self.n_rec_article
        #  varables for precision and recall
        #记录推荐正确的电影数
        hit = 0
        #记录推荐电影的总数
        rec_count = 0
        #记录测试数据中总数
        test_count = 0
        # varables for coverage
        all_rec_articles = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_articles = self.testset.get(user, {})
            rec_articles = self.recommend(user)
            for article, _ in rec_articles:
                if article in test_articles:
                    hit += 1
                all_rec_articles.add(article)
                popular_sum += math.log(1 + self.article_popular[article])
            rec_count += N
            test_count += len(test_articles)
        # 计算准确度
        precision = hit / (1.0 * rec_count)
        # 计算召回率
        recall = hit / (1.0 * test_count)
        # 计算覆盖率
        coverage = len(all_rec_articles) / (1.0 * self.article_count)
        #计算流行度
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)



ratingfile = 'data/ods_sql_lfm_postive_data.csv'
usercf = UserBasedCF()
usercf.generate_dataset(ratingfile)
# if not os.path.exists('cf.model'):
usercf.calc_user_sim()
# usercf.calc_user_sim()

print(' usercf.users, len=',len(usercf.users),' ', usercf.users[:10])
print(' usercf.items,len=',len(usercf.items),' ',usercf.items[:10])
# for user in usercf.users:
#     a = usercf.recommend(user)

# print(a)
# '''
# 以下为用户id为100的用户推荐的资讯
# a = usercf.recommend("100")
# cost time: 2.384185791015625e-05
# [(186, 0.5), (128, 0.5), (108, 0.5), (117, 0.4472135954999579), (140, 0.4472135954999579), (104, 0.4472135954999579), (106, 0.4472135954999579)]
# '''
# usercf.evaluate()
