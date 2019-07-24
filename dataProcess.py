#-*- coding: utf-8 -*-
'''
Created on 2019-07-13
To process data and build feature input for model training
@author: yuanfang
'''


from sklearn.utils import shuffle
import happybase
from odps import ODPS
from utils import Utils
import time
import pandas as pd


def metric_time(f):
    def wrapper(df,column):
        start_time = time.time()
        df=f(df,column)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        print("time is %d ms" %execution_time )
        return df
    return wrapper

class DataProcess(object):

    def __init__(self):
        self.table_name='RECOMMEND_MATRIX_test'
        self.host="cdh06.ali.aiwaystack.com"
        self.port=9090
        self.tables=[
                'ods_app_ai_content_content_news_all_dt',
                'ods_app_ai_aiid_account_all_dt',
                'ods_app_ai_content_content_topic_all_dt',
                'dwd_user_app_browse_action_detail_cdc_dt',

                'ods_app_ai_content_content_news_comment_all_dt',
                'ods_app_ai_content_content_news_comment_liked_all_dt',
                'ods_app_ai_content_content_news_liked_all_dt',
                'ods_app_ai_content_content_topic_comment_all_dt',
                'ods_app_ai_content_content_topic_comment_liked_all_dt',
                'ods_app_ai_content_content_topic_discuss_all_dt',
                'ods_app_ai_content_content_topic_discuss_comment_all_dt',
                'ods_app_ai_content_content_topic_discuss_liked_all_dt',
                'ods_app_ai_content_content_topic_liked_all_dt',
                'ods_app_ai_content_content_topic_vote_all_dt',
                'ods_app_ai_content_content_topic_vote_question_all_dt',
                'ods_app_ai_content_content_topic_vote_option_all_dt',
                'ods_app_ai_content_content_topic_vote_option_user_all_dt'
                ]
        self.utils=Utils()




    def toHabse(self,usercf):
        """
         store data into Hbase
         host：cdh06.ali.aiwaystack.com,cdh05.ali.aiwaystack.com,cdh04.ali.aiwaystack.com:9090,  2181 for zookeeper
         table：RECOMMEND_MATRIX
        """
        start_processing=time.time()
        conn = happybase.Connection(self.host,self.port)#如果没有，则开启thrift:`hbase thrift start
        try:
            conn.disable_table(self.table_name)
            conn.delete_table(self.table_name)
        except :
            print(' '+self.table_name+' table existes')
            # IOError
        families = {
        'user': dict(max_versions=10),
        'article': dict(max_versions=10)
        }
        conn.create_table(self.table_name, families)
        table = conn.table(self.table_name)

        with table.batch(batch_size=10) as bat:
            index=0
            for userid in usercf.users:
                recommends = usercf.recommend(userid)
                for article_id, score in recommends:
                    index += 1
                    bat.put(str(index), {'user:user_id':str(userid),'article:article_id': str(article_id),'article:article_Rscore': str(score),'article:article_type': str(usercf.items[article_id])})
        print('finished -------- save data into Habse. ',time.time()-start_processing,'s')

    def _gettables(self):
        """
        Get all data from ali ods.
        """
        print('Downloading tables from ods sql.')
        start_processing=time.time()
        o = ODPS('LTAIVYhmNLQm0RPD', 'C7mhqOapX1iUSCYwis3lrZFN16nX5x', 'WS_BigData',
                    endpoint='http://service.cn.maxcompute.aliyun.com/api')

        for table in self.tables:
            pre_time=time.time()
            print(table)
            table_data = o.get_table(table).to_df().to_pandas()
            print("connect ",table," successfully  cost",time.time()-pre_time)
            pre_time=time.time()
            table_data.to_csv('data/'+table+'.csv')
            print("save ",table," successfully  cost",time.time()-pre_time)
        print('finished --------Downloading tables from ods sql.',time.time()-start_processing)


    #--------------just save a single partition
        print('Fetch specific pt from downloaded tables.')
        fetch_pt_time=time.time()
        tables=[
                'ods_app_ai_content_content_news_all_dt',
                'ods_app_ai_aiid_account_all_dt',
                'ods_app_ai_content_content_topic_all_dt',
                # 'dwd_user_app_browse_action_detail_cdc_dt',
                'ods_app_ai_content_content_news_comment_all_dt',
                'ods_app_ai_content_content_news_comment_liked_all_dt',
                'ods_app_ai_content_content_news_liked_all_dt',
                'ods_app_ai_content_content_topic_comment_all_dt',
                'ods_app_ai_content_content_topic_comment_liked_all_dt',
                'ods_app_ai_content_content_topic_discuss_all_dt',
                'ods_app_ai_content_content_topic_discuss_comment_all_dt',
                'ods_app_ai_content_content_topic_discuss_liked_all_dt',
                'ods_app_ai_content_content_topic_liked_all_dt',
                'ods_app_ai_content_content_topic_vote_all_dt',
                'ods_app_ai_content_content_topic_vote_question_all_dt',
                'ods_app_ai_content_content_topic_vote_option_all_dt',
                'ods_app_ai_content_content_topic_vote_option_user_all_dt'
                ]
        for table in tables:
            pre_time=time.time()
            print(table+'.csv')
            table_data = pd.read_csv('data/'+table+'.csv',index_col=0,dtype=str)
            print("read ",table,"   cost",time.time()-pre_time)
            pre_time=time.time()
            table_data[table_data['pt'] =='20190717'].to_csv('data/pt20190717'+table+'.csv')
            print("save pt20190717-",table," successfully  cost",time.time()-pre_time)
        print('finished --------Fetch specific pt from downloaded tables.',time.time()-fetch_pt_time)

    def _cleanDirtydata(self):
        """
        Clean data in tables .
        """

    #--------------cleaning account table
        account_time=taccount=time.time()
        print('Cleaning account table.')
        accountdf=pd.read_csv('data/pt20190717ods_app_ai_aiid_account_all_dt.csv')#v,index_col=0不能制定index，否这无法增加列

        accountdf.dropna(axis=0,subset = ["id"])   # 丢弃‘userid’这两列中有缺失值的行
        accountdf['nickname'][accountdf['nickname'].isnull()]=0
        accountdf['nickname'][accountdf['nickname'].isnull()!=True]=1
        accountdf['name'][accountdf['name'].isnull()!=True]=1
        accountdf['name'][accountdf['name'].isnull()]=0
        accountdf['sex'][accountdf['sex'].isnull()!=True]=1
        accountdf['sex'][accountdf['sex'].isnull()]=0
    #   accountdf['reg_time'][accountdf['reg_time'].isnull()]='0'
        # t=time.time()
        # accountdf=decompose_time('reg_time',accountdf)
        pre=time.time()
        # print('accout decompose_time cost:',pre-t)
        accountdf['id_card'][accountdf['id_card'].isnull()!=True]=1
        accountdf['id_card'][accountdf['id_card'].isnull()]=0
        accountdf['head_url'][accountdf['head_url'].isnull()!=True]=1
        accountdf['head_url'][accountdf['head_url'].isnull()]=0
        accountdf['owner_type'][accountdf['owner_type'].isnull()!=True]=1
        accountdf['owner_type'][accountdf['owner_type'].isnull()]=0
        accountdf['age']=self.utils.birthday_to_age(accountdf['birthday'])

        print('accout birthday_to_age cost:',time.time()-pre)
        pre=time.time()
        accountdf['region'][accountdf['region'].isnull()]='0'
        accountdf['region']=self.utils.category_to_number(accountdf.region.values, 'region')

        print('accout category_to_number cost:',time.time()-pre)
        pre=time.time()
        def p(x):
            print("column:",x)
            print("is not null: ",accountdf[x][accountdf[x].isnull()!=True].to_list())
            print("sum=",sum(accountdf[x][accountdf[x].isnull()!=True].to_list()))
            print("count(owner_type)=",len(accountdf[x][accountdf[x].isnull()!=True].to_list()))
            print("min=",min(accountdf[x][accountdf[x].isnull()!=True].to_list()))
            print("max=",max(accountdf[x][accountdf[x].isnull()!=True].to_list()))
        # p('region')
        accountdf['integral'].fillna(0,inplace=True)
        accountdf['integral']=accountdf['integral'].apply(lambda x:0 if x<0 else x )
        # p('integral')
        accountdf['is_big_v'].fillna(0,inplace=True)
        # p('is_big_v')
        accountdf['is_first_login'].fillna(0,inplace=True)
        accountdf['is_first_login']=accountdf['is_first_login'].astype('int')
        accountdf['autograph'][accountdf['autograph'].isnull()!=True]=1
        accountdf['autograph'][accountdf['autograph'].isnull()]=0
        # p('autograph')
        accountdf['user_type'][accountdf['user_type'].isnull()!=True]=1
        accountdf['user_type'][accountdf['user_type'].isnull()]=0
        accountdf['event'][accountdf['event'].isnull()!=True]=1
        accountdf['event'][accountdf['event'].isnull()]=0
        accountdf['agreement_version'][accountdf['agreement_version'].isnull()!=True]=1
        accountdf['agreement_version'][accountdf['agreement_version'].isnull()]=0
        accountdf['big_region'][accountdf['big_region'].isnull()!=True]=1
        accountdf['big_region'][accountdf['big_region'].isnull()]=0

        print('accout others cost:',time.time()-pre)
        pre=time.time()
        # p('big_region')# value is empty
        towrite=accountdf.loc[:,['id','nickname','name','sex','id_card','head_url','owner_type','age','region','integral','is_big_v',
                         'is_first_login','autograph','user_type','event','agreement_version','big_region']]

        # print('accout loc cost:',time.time()-pre)
        towrite.dropna(axis=0, how='any', inplace=True)
        print('accout drop cost:',time.time()-pre)
        pre=time.time()
        print('account table cost in total:',pre-taccount)
        towrite.to_csv("data/clean_sql_account.csv")
        print('finished --------Cleaning account table.',time.time()-account_time)

    #--------------cleaning news table
    # 'id', 'creator_id', 'created_time', 'updator_id',
    #        'updated_time', 'delete_flag', 'news_title', 'news_subtitle',
    #        'news_source', 'news_summary', 'img_url', 'liked_num', 'clap_num',
    #        'recommend_flag', 'sort_no', 'publish_flag', 'news_detail',
    #        'comment_num', 'read_num', 'news_type', 'follow_num', 'share_num',
    #        'status', 'author', 'publish_time', 'video_url', 'pt'
        print('Cleaning news table.')
        newstable=time.time()
        newdf=pd.read_csv('data/pt20190717ods_app_ai_content_content_news_all_dt.csv')#v,index_col=0不能制定index，否这无法增加列
        newdf.dropna(axis=0,subset = ["id"])   # 丢弃‘userid’这两列中有缺失值的行

    #     print('news loc cost:',time.time()-pre)
        pre=time.time()
        # print(accountdf.columns)
        # print(accountdf.describe())
        # print(accountdf.isnull().any()) #which column is null

        count=newdf['id'].value_counts()
        # if count.values.sum() == len(count):
        #     print("No duplicated data in the frame， length=",len(count))
        # else:
        #     print("you need to go back for cleaning data ")
        print(newdf.isnull().any()) #which column is null
        newdf['creator_id'].fillna(0,inplace=True)
        # newdf['updated_time'][newdf['updated_time'].isnull()]='0'
        # newdf = decompose_time('updated_time',newdf)
        # print('news decompose_time cost:',time.time()-pre)
        pre=time.time()
        newdf['news_title']=newdf['news_title'].apply(lambda x:1 if isinstance(x,str) else 0)
        newdf['news_subtitle']=newdf['news_subtitle'].apply(lambda x:1 if isinstance(x,str) else 0)

        print('news news_title news_subtitle cost:',time.time()-pre)
        pre=time.time()
        # print(newdf.isnull().any()) #判断那一列有空
        def removeItems():
            flags =newdf['delete_flag'].tolist()
            removeIndexs=[]
            for index, flag in enumerate(flags):
                if flag == 1:
                    removeIndexs.append(index)
            # print("delete:",removeIndexs[:-1])

            flags1 =newdf['publish_flag'].tolist()
            for index1, flag1 in enumerate(flags1):
                if flag1 == 0:
                    # print("unpublish index=",index1)
                    removeIndexs.append(index1)
            # print("remove news:",removeIndexs[:-1])
            newdf.drop(newdf.index[removeIndexs],inplace=True)

        removeItems()

        print('news removeItems cost:',time.time()-pre)
        pre=time.time()
    #     newdf['newsid']=newdf['id']
        # df_toWrite=newdf.loc[:,['creator_id','news_title','news_subtitle',
        #                         'updated_year','updated_month','updated_day',
        #                         'updated_timestamp','newsid']]
        df_toWrite1=newdf.loc[:,['id','creator_id','news_title','news_subtitle','news_type']]
        print('news loc cost:',time.time()-pre)
        pre=time.time()
        df_toWrite1.dropna(axis=0, how='any', inplace=True)
        print('news dropna cost:',time.time()-pre)
        pre=time.time()
        df_toWrite1.to_csv("data/clean_sql_news.csv")
        print('news to_csv cost:',time.time()-pre)
        pre=time.time()
        print('finished --------Cleaning news table.', newstable-time.time())

    # --------------cleaning click table
        print('Cleaning click table.')
        clicktable=time.time()
        clickdf=pd.read_csv('data/dwd_user_app_browse_action_detail_cdc_dt.csv')#v,index_col=0不能制定index，否这无法增加列
        clickdf.drop(clickdf['article_id'][clickdf['article_id'].isnull()].index.to_list(), inplace=True)
        clickdf.drop(clickdf['user_id'][clickdf['user_id'].isnull()].index.to_list(), inplace=True)
        print(' cleaning, Drop null of user_id, then the length of news table=',len(clickdf))
        # clickdf=clickdf.loc[:5000,:]
        def remove_useless(df,column):
            flags =df[column].tolist()
            removeIndexs=[]
            for index, flag in enumerate(flags):
                if flag == 'unlogin':
                    removeIndexs.append(index)
            # flags1 =clickdf['article_id'].tolist()
            # for index1, flag1 in enumerate(flags1):
            #     if flag1 == '\\N':
            #         removeIndexs.append(index1)
            df.drop(df.index[removeIndexs],inplace=True)
            return df
        print('before len=',len(clickdf))
        clickdf=remove_useless(clickdf,'user_id')
        print('after len=',len(clickdf))

        @metric_time
        def mixTypetoint(df,column):
            for index in df[column].index:
                try:
                    df.loc[index,column]=int(float(df.loc[index,column]))
                except ValueError:
                    print("Error: ValueError article_id= ",df.loc[index,column],'index=',index," and give article_id to 0")
                    df.loc[index,column]=0
            return df
        print("len of null -article-id",len(clickdf['article_id'][clickdf['article_id'].isnull()].index.to_list()))
        # print('before len=',len(clickdf))
        clickdf=mixTypetoint(clickdf,'article_id')
        # print('after len=',len(clickdf))

        df_toWrite=clickdf.loc[:,['user_id','article_id']]
        df_toWrite.dropna(axis=0, how='any', inplace=True)
        print('after len=',len(df_toWrite))

        df_toWrite.to_csv("data/clean_sql_clicked.csv")
        print('cleaning clicked table cost:',time.time()-clicktable)
        print('finished --------Cleaning click table.', account_time-time.time())
        # print(clickdf.columns)
        # print(clickdf.describe())
        # print(clickdf.isnull().any()) #which column is null


    def featureBuilding(self):
        # --------------merge tables into training set
        # self._gettables()
        # self._cleanDirtydata()
        print('Build features')
        mergetime=pre_time=time.time()
        df_account=pd.read_csv("data/clean_sql_account.csv")
        df_news=pd.read_csv("data/clean_sql_news.csv")
        df_clicked=pd.read_csv("data/clean_sql_clicked.csv",dtype=int)
        df_account.rename(columns={'id':'userid'},inplace=True)
        df_news.rename(columns={'id':'newsid'},inplace=True)
        df_clicked.rename(columns={'user_id':'userid','article_id':'newsid'},inplace=True)
        # df_clicked.rename(columns={'user_id':'userid'},inplace=True)
        df_account.drop(df_account.columns[0], axis=1, inplace=True)       # 删除第1列
        df_news.drop(df_news.columns[0], axis=1, inplace=True)       # 删除第1列
        df_clicked.drop(df_clicked.columns[0], axis=1, inplace=True)       # 删除第1列

        print("clicktbale  ",df_clicked.columns, "table len=",len(df_clicked),"user number=",len(df_clicked.userid.unique()))
        print('----------'*10)
        print("usertable  ",df_account.columns, "table len=",len(df_account),"user number=",len(df_account.userid.unique()))
        print('----------'*10)
        print("newtable  ",df_news.columns, "table len=",len(df_news),"news number=",len(df_news.newsid.unique()))
        print('----------'*10)

        pos_merge1=pd.merge(df_clicked,df_account,on='userid',how='inner')
        # print("pos_merge1 ",pos_merge1.columns, "table len=",len(pos_merge1),"userid number=",len(pos_merge1.userid.unique()))
        # print('----------'*10)

        pos_merge2=pd.merge(pos_merge1,df_news,on='newsid',how='inner')
        # print("pos_merge2 ",pos_merge2.columns, "table len=",len(pos_merge2),"userid number=",len(pos_merge2.userid.unique()))
        # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
        pos_merge2['label']=1
        pos_merge2.dropna(axis=0, how='any', inplace=True)
        # pos_merge2.to_csv('data/ods_sql_postive_data.csv')
        pos_merge2['useridReindex']=self.utils.category_to_number_onNewColumn(pos_merge2.userid, 'userid')
        pos_merge2['newsidReindex']=self.utils.category_to_number_onNewColumn(pos_merge2.newsid, 'newsid')
        path='data/ods_sql_lfm_postive_data.csv'
        pos_merge2.to_csv(path)
        print('build ods_sql_lfm_postive_data table, cost',time.time()-pre_time)
        print('finished --------Build features.', time.time()-mergetime)
        print('build ods_sql_postive_data, cost',time.time()-pre_time)
        pre_time=time.time()
        length=len(pos_merge2)

        negative_users=set(df_account['userid'].to_list())^set(df_clicked['userid'].tolist())
        print("negative_articles len=",len(negative_users)," length=",length)

        negative_clickdf=pd.DataFrame({'userid':list(negative_users)[:length],'newsid':df_clicked['newsid'].to_list()[:length]})

        neg_merge1=pd.merge(negative_clickdf,df_account,on='userid',how='inner')
        # print("merged1 ",neg_merge1.columns, "table len=",len(neg_merge1),"userid number=",len(neg_merge1.userid.unique()))
        # print('----------'*10)
        neg_merge2=pd.merge(neg_merge1,df_news,on='newsid',how='inner')
        # print("merged2 ",neg_merge2.columns, "table len=",len(neg_merge2),"userid number=",len(neg_merge2.userid.unique()))
        # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
        neg_merge2['label']=0
        neg_merge2.dropna(axis=0, how='any', inplace=True)
        neg_merge2.to_csv('data/ods_sql_negative_data.csv')
        print('build ods_sql_negative_data, cost',time.time()-pre_time)
        pre_time=time.time()

        trainingdf=shuffle(pd.concat([neg_merge2,pos_merge2]))
        # print("columns0 ",trainingdf.columns)
        # trainingdf.reset_index()
        # print("columns1 ",trainingdf.columns)
        # trainingdf.drop('index', axis=1, inplace=True)       # 删除第1列
        print(trainingdf.columns)
        trainingdf.to_csv('data/ods_sql_traningset.csv')
        print('build ods_sql_traningset, cost,',time.time()-pre_time)
        print('finished --------Feature building takes.', time.time()-mergetime,"s in total.")
        return path
