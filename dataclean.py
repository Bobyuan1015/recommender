# To do a feature engineering for recommendation system
# Author:yuanfang
# Date:2019-07-13

import pandas as pd
import numpy as np
from pandas import DataFrame
import time
import datetime
from sklearn.utils import shuffle
import os
import datetime as dt
import math

from odps import ODPS

import linecache


def metric_time(f):
    def wrapper(df,column):
        start_time = time.time()
        df=f(df,column)
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        print("time is %d ms" %execution_time )
        return df
    return wrapper


def decompose_time(name,newdf):
    previous_time= time.time()
    toReplace1=newdf[name][newdf[name].isnull()].index.to_list()
    datetimeList=[]
    print("decompose time index.to_list() cost",time.time()-previous_time)
    previous_time= time.time()
    for index1, flag1 in enumerate(newdf[name].tolist()):
        # print("index=",index1," content:",flag1)
        if flag1 == '0' or isinstance(flag1,float):
            toReplace1.append(index1)
        else:
            if isinstance(newdf[name][index1],str) !=True:
                print(newdf[name][index1])
            newdf[name][index1]=string_toDatetime(newdf[name][index1])
            datetimeList.append(newdf[name][index1])
    print("decompose time replacement cost",time.time()-previous_time)
    previous_time= time.time()
    minCost=datetimeList[0]
    # minCost=min(datetimeList)
    for index in toReplace1:
        newdf[name][index]=minCost
    print("decompose time min() cost",time.time()-previous_time)
    previous_time= time.time()
    # for index,_ in enumerate(newdf['end_time']):
    #     print("end_time",type(newdf.start_time[index])," replaced ",newdf.start_time[index])

    # print(" length of null starttime is :",len(newdf.start_time[newdf['start_time'].isna() != False]))
    dates = pd.to_datetime(newdf[name], format="%Y-%m-%d %H:%M:%S")
    newdf[name+'_year']=dates.dt.year
    newdf[name+'_month']=dates.dt.month
    newdf[name+'_week']=dates.dt.week
    newdf[name+'_day']=dates.dt.day
    newdf[name+'_timestamp'] = newdf[name].apply(lambda x:string_toTimestamp(datetime_toString(x)))
    print("decompose time  year month day week cost",time.time()-previous_time)
    previous_time= time.time()
    # print("replaced: ",newdf['end_time_timestamp'])
    def minMaxNormalization(x):
        return DataFrame({"updated_timestamp":np.rint([1000*(float(i)-min(x))/float(max(x)-min(x)) for i in x])})
    newdf[name+'_timestamp']=minMaxNormalization(newdf[name+'_timestamp'])
    print("decompose time  minMaxNormalization cost",time.time()-previous_time)

    return newdf

def justSaveNewsDetail():
    print('just save news content into a file')
    df=pd.read_csv('data/new-temp_user_news.csv',index_col=0)
    print(df.columns)
    print(df['news_detail'][df['news_detail'].isnull()].index.to_list())
    # df_toWrite=df.loc[:,['news_id','news_title', 'news_subtitle', 'news_detail']]
    # print(df_toWrite.columns)
    # df_toWrite.to_csv('news.csv')
    # df=pd.read_csv('news.csv',index_col=0)
    # print(df.columns)



def removeTapIn_oneColumn(filename):
    linecache.clearcache()
    fp = open(filename, 'r')
    paths=filename.split('/')
    path=''.join(paths[:-1])
    old_name=''.join(paths[-1:])
    fp1=open(path+'/mid-'+old_name,'wb')
    columns=linecache.getline(filename, 1).strip().split(':!:')
    print( columns)
    dilimeterNumbers=len(columns)-1
    tmp=['']
    for i, line in enumerate(fp):

            num = (len(line) - len(line.replace(':!:',""))) // len(':!:')
            if num == dilimeterNumbers :
                # print(' write:')
                fp1.write(bytes(line,'UTF-8'))
            else:
                tmp[len(tmp)-1]+=line.strip('\r\n')
                tempnum = (len(tmp[len(tmp)-1]) - len(tmp[len(tmp)-1].replace(':!:',""))) // len(':!:')
                if tempnum ==dilimeterNumbers:
                    fp1.write(bytes(tmp[len(tmp)-1],'UTF-8'))
                    tmp.append('')
                    # print("append tempnum=",tempnum,'----len of tmp=',len(tmp[len(tmp)-1]),' index=',len(tmp)-1,' ',tmp[len(tmp)-1])
    print("converting job is done, you can check the file.")
    fp1.close()
    fp.close()

    f=open(path+'/mid-'+old_name,'r')
    dfDic={}
    for i,column in enumerate(columns):
        dfDic.setdefault(column, [])

    for index, line in enumerate(f):
        if index!=0:
            # print("index=",index," line=",line)
            words= line.split(':!:')
            for i,column in enumerate(columns):
                dfDic[column].append(words[i].strip())

    df=pd.DataFrame(dfDic)
    # df.to_csv('newcorrct_news_title_comment_like.csv',sep=',')
    df.to_csv(path+'/new-'+old_name.split('.')[0]+'.csv',sep=',')
    # f=open(path+'/mid'+paths[-2:-1],'r')
    # df.to_table('test.txt', sep=':!:')
    df1=pd.read_csv(path+'/new-'+old_name.split('.')[0]+'.csv', sep=',',index_col=0)
    print(df1.head(15),"  len=",len(df1))
    return path+'/new-'+old_name.split('.')[0]+'.csv'



def cleanData():

#--------------get all data from ali ods
    print('Downloading tables from ods sql.')
    start_processing=time.time()
    o = ODPS('LTAIVYhmNLQm0RPD', 'C7mhqOapX1iUSCYwis3lrZFN16nX5x', 'WS_BigData',
                endpoint='http://service.cn.maxcompute.aliyun.com/api')
    tables=[
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
    for table in tables:
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

        # table_data[table_data['pt'] ==20190717].to_csv('ods/pt20190717'+table+'.csv')
        print("save pt20190717-",table," successfully  cost",time.time()-pre_time)
    print('finished --------Fetch specific pt from downloaded tables.',time.time()-fetch_pt_time)
#--------------cleaning account table
    account_time=taccount=time.time()
    print('Cleaning account table.')
    accountdf=pd.read_csv('data/pt20190717ods_app_ai_aiid_account_all_dt.csv')#v,index_col=0不能制定index，否这无法增加列

    accountdf.dropna(axis=0,subset = ["id"])   # 丢弃‘userid’这两列中有缺失值的行
    # print(accountdf.columns)
    # print(accountdf.describe())
    # print(accountdf.isnull().any()) #which column is null
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
    accountdf['age']=birthday_to_age(accountdf['birthday'])

    print('accout birthday_to_age cost:',time.time()-pre)
    pre=time.time()
    accountdf['region'][accountdf['region'].isnull()]='0'
    accountdf['region']=category_to_number(accountdf.region.values,'region')

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
    # towrite=accountdf.loc[:,['id','nickname','name','sex','reg_time_year','reg_time_month','reg_time_week','reg_time_day',
    #                  'reg_time_timestamp','id_card','head_url','owner_type','age','region','integral','is_big_v',
    #                  'is_first_login','autograph','user_type','event','agreement_version','big_region']]
    towrite=accountdf.loc[:,['id','nickname','name','sex','id_card','head_url','owner_type','age','region','integral','is_big_v',
                     'is_first_login','autograph','user_type','event','agreement_version','big_region']]

    # print('accout loc cost:',time.time()-pre)
    towrite.dropna(axis=0, how='any', inplace=True)
    print('accout drop cost:',time.time()-pre)
    pre=time.time()
    print('account table cost in total:',pre-taccount)
    towrite.to_csv("data/clean_sql_account.csv")
    print('finished --------Cleaning account table.',time.time()-account_time)
# accountdf['d_type'][accountdf['d_type'].isnull()]=0
    # accountdf['d_type'][accountdf['d_type'].isnull()!=True]=1
    # accountdf['device_token'][accountdf['device_token'].isnull()]=0
    # accountdf['device_token'][accountdf['device_token'].isnull()!=True]=1

    # accountdf['license_time'][accountdf['license_time'].isnull()]=0
    # accountdf['license_time'][accountdf['license_time'].isnull()!=True]=1
  # p('is_first_login')
    # # accountdf['address'][accountdf['address'].isnull()]=0
    # # accountdf['address'][accountdf['address'].isnull()!=True]=1
 # p('user_type')
    # accountdf['empirical'][accountdf['empirical'].isnull()!=True]=1
    # accountdf['empirical'][accountdf['empirical'].isnull()]=0
    # p('empirical')# value is empty
 # p('event')
    # accountdf['company'][accountdf['company'].isnull()!=True]=1
    # accountdf['company'][accountdf['company'].isnull()]=0
    # p('company')# value is empty
    # accountdf['drive_license'][accountdf['drive_license'].isnull()!=True]=1
    # accountdf['drive_license'][accountdf['drive_license'].isnull()]=0
    # p('drive_license')# value is empty
    # accountdf['security_state'][accountdf['security_state'].isnull()!=True]=1
    # accountdf['security_state'][accountdf['security_state'].isnull()]=0
    # p('agreement_version')
    # accountdf['email'][accountdf['email'].isnull()!=True]=1
    # accountdf['email'][accountdf['email'].isnull()]=0
    # p('email')# value is empty

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

    print('news loc cost:',time.time()-pre)
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
    df_toWrite1=newdf.loc[:,['id','creator_id','news_title','news_subtitle']]
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
        # print(removeIndexs[:20],len(newdf.user_id.unique()))
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
    print('before len=',len(clickdf))
    clickdf=mixTypetoint(clickdf,'article_id')
    print('after len=',len(clickdf))

    df_toWrite=clickdf.loc[:,['user_id','article_id']]
    # print('after cleaning, the length of news table=',len(df_toWrite),'newsid:',df_toWrite['newsid'].to_list()[:10]
    #       ,clickdf['newsid'].to_list()[:10])
    df_toWrite.dropna(axis=0, how='any', inplace=True)
    print('after len=',len(df_toWrite))

    df_toWrite.to_csv("data/clean_sql_clicked.csv")
    print('cleaning clicked table cost:',time.time()-clicktable)
    pre=time.time()
    print('after len=',len(df_toWrite))
    print('finished --------Cleaning click table.', clicktable-time.time())
    # print(clickdf.columns)
    # print(clickdf.describe())
    # print(clickdf.isnull().any()) #which column is null

# --------------merge tables into training set

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
    print("pos_merge1 ",pos_merge1.columns, "table len=",len(pos_merge1),"userid number=",len(pos_merge1.userid.unique()))
    print('----------'*10)


    pos_merge2=pd.merge(pos_merge1,df_news,on='newsid',how='inner')
    print("pos_merge2 ",pos_merge2.columns, "table len=",len(pos_merge2),"userid number=",len(pos_merge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    pos_merge2['label']=1
    pos_merge2.dropna(axis=0, how='any', inplace=True)
    pos_merge2.to_csv('data/ods_sql_postive_data.csv')
    print('build ods_sql_postive_data, cost',time.time()-pre_time)
    pre_time=time.time()
    length=len(pos_merge2)


    # negative_articles=set(df_news['newsid'].to_list())^set(df_clicked['newsid'].tolist())
    # print("negative_articles len=",len(negative_articles)," length=",length)

    # negative_clickdf=pd.DataFrame({'userid':df_clicked['userid'].to_list()[:length],'newsid':list(negative_articles)[:length]})
    negative_users=set(df_account['userid'].to_list())^set(df_clicked['userid'].tolist())
    print("negative_articles len=",len(negative_users)," length=",length)

    negative_clickdf=pd.DataFrame({'userid':list(negative_users)[:length],'newsid':df_clicked['newsid'].to_list()[:length]})

    neg_merge1=pd.merge(negative_clickdf,df_account,on='userid',how='inner')
    print("merged1 ",neg_merge1.columns, "table len=",len(neg_merge1),"userid number=",len(neg_merge1.userid.unique()))
    print('----------'*10)
    neg_merge2=pd.merge(neg_merge1,df_news,on='newsid',how='inner')
    print("merged2 ",neg_merge2.columns, "table len=",len(neg_merge2),"userid number=",len(neg_merge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    neg_merge2['label']=0
    neg_merge2.dropna(axis=0, how='any', inplace=True)
    neg_merge2.to_csv('data/ods_sql_negative_data.csv')
    print('build ods_sql_negative_data, cost',time.time()-pre_time)
    pre_time=time.time()

    trainingdf=shuffle(pd.concat([neg_merge2,pos_merge2]))
    print("columns0 ",trainingdf.columns)
    trainingdf.reset_index()
    print("columns1 ",trainingdf.columns)
    # trainingdf.drop('index', axis=1, inplace=True)       # 删除第1列
    print(trainingdf.columns)
    trainingdf.to_csv('data/ods_sql_traningset.csv')
    print('build ods_sql_traningset, cost,',time.time()-pre_time)
    pre_time=time.time()

    pos_merge2['useridReindex']=category_to_number_onNewColumn(pos_merge2.userid,'userid')
    pos_merge2['newsidReindex']=category_to_number_onNewColumn(pos_merge2.newsid,'newsid')
    pos_merge2.to_csv('data/ods_sql_lfm_postive_data.csv')
    print('build ods_sql_lfm_postive_data table, cost',time.time()-pre_time)
    print('finished --------Build features.', time.time()-mergetime)
    print('finished --------Feature engineering takes.', time.time()-start_processing," in total.")

#建立单个文件的excel转换成csv函数,file 是excel文件名，to_file 是csv文件名。
def excel_to_csv(file,to_file):
    data_xls=pd.read_excel(file,sheet_name=0)
    data_xls.to_csv(to_file,encoding='utf_8_sig')


#读取一个目录里面的所有文件：
def read_path(path):
    dirs=os.listdir(path)
    return dirs

def category_to_number(x,key):
        listedArray = list(x)
        for index,i in enumerate(x):
            if isinstance(i,str) !=True:
                listedArray[index]='0'
        dic={}
        for index,item in enumerate(np.unique(listedArray)):
            dic[item]=index
        for st_index, i in  enumerate(listedArray):
            listedArray[st_index]=dic[i]
        return DataFrame({key:listedArray})

def birthday_to_age(x):
        listedArray = list(x)
        now_year=dt.datetime.today().year #当前的年份
        for index,item in enumerate(listedArray):
            # print("out item:",item)
            # print("out item: type=",type(item))
            if isinstance(item,float):
                # print("inner ",item)
                if math.isnan(item):
                    # print("give it to 0")
                    listedArray[index]=0 #could be improved by mean()
            else:
                # print("inner item:",item)
                # print("inner item: type=",type(item))
                listedArray[index]=now_year-datetime.datetime.strptime(item,"%Y-%m-%d %H:%M:%S").year

        return pd.Series(listedArray)

def category_to_number_onNewColumn(x,column):
        listedArray = list(x)
        dic={}
        for index,item in enumerate(np.unique(x)):
            dic[item]=index
        for st_index, i in  enumerate(listedArray):
            listedArray[st_index]=dic[i]
        return DataFrame({column+'Reindex':listedArray})

def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")
def string_toDatetime(st):
    return datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
def string_toTimestamp(st):
    return time.mktime(time.strptime(st, "%Y-%m-%d %H:%M:%S"))


def decode_time(newdf,column):
    if column == 'start_time':
        # for index,_ in enumerate(newdf['start_time']):
        #         print("start_time",type(newdf.start_time[index])," old ",newdf.start_time[index])
        toReplace=newdf['start_time'][newdf['start_time'].isnull()].index.to_list()#replace null of start_time with create_time
        for index1, flag1 in enumerate(newdf['start_time'].tolist()):
            if flag1 == '\\N':
                toReplace.append(index1)
        for index in toReplace:
            newdf.start_time[index]=newdf.created_time[index]

        # for index,_ in enumerate(newdf['start_time']):
        #     print("start_time",type(newdf.start_time[index])," replaced ",newdf.start_time[index])
        # print(" length of null starttime is :",len(newdf.start_time[newdf['start_time'].isna() != False]))
        dates1 = pd.to_datetime(newdf['start_time'], format="%Y-%m-%d %H:%M:%S")
        newdf['start_time_year']=dates1.dt.year
        newdf['start_time_month']=dates1.dt.month
        newdf['start_time_week']=dates1.dt.week
        newdf['start_time_day']=dates1.dt.day

        for index,_ in enumerate(newdf['start_time']):
            if isinstance(newdf['start_time'][index],str) != True:
                newdf['start_time'][index]=datetime_toString(newdf['start_time'][index])#.strftime("%Y-%m-%d %H:%M:%S")
            # print(type(newdf.start_time[index])," ",newdf.start_time[index])

        newdf['start_time_timestamp']=newdf['start_time'].apply(lambda x:string_toTimestamp(x))
        # print(" ",newdf['start_time_timestamp'])
        def minMaxNormalization(x):
            return DataFrame({"start_time_timestamp":np.rint([10000*(float(i)-min(x))/float(max(x)-min(x)) for i in x])})
        newdf['start_time_timestamp']=minMaxNormalization(newdf['start_time_timestamp'])

    elif column == 'end_time':
        # for index,_ in enumerate(newdf['end_time']):
        #         print("end_time",type(newdf.end_time[index])," old ",newdf.end_time[index])
        toReplace1=newdf['end_time'][newdf['end_time'].isnull()].index.to_list()
        datetimeList=[]
        for index1, flag1 in enumerate(newdf['end_time'].tolist()):
            if flag1 == '\\N':
                toReplace1.append(index1)
            else:
                newdf['end_time'][index1]=string_toDatetime(newdf['end_time'][index1])
                datetimeList.append(newdf['end_time'][index1])
        for index in toReplace1:
            newdf.end_time[index]=max(datetimeList)
        # for index,_ in enumerate(newdf['end_time']):
        #     print("end_time",type(newdf.start_time[index])," replaced ",newdf.start_time[index])

        # print(" length of null starttime is :",len(newdf.start_time[newdf['start_time'].isna() != False]))
        dates = pd.to_datetime(newdf['end_time'], format="%Y-%m-%d %H:%M:%S")
        newdf['end_time_year']=dates.dt.year
        newdf['end_time_month']=dates.dt.month
        newdf['end_time_week']=dates.dt.week
        newdf['end_time_day']=dates.dt.day
        newdf['end_time']=newdf['end_time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        newdf['end_time_timestamp'] = newdf['end_time'].apply(lambda x:time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))
        # print("replaced: ",newdf['end_time_timestamp'])
        def minMaxNormalization(x):
            return DataFrame({"updated_timestamp":np.rint([1000*(float(i)-min(x))/float(max(x)-min(x)) for i in x])})
        newdf['end_time_timestamp']=minMaxNormalization(newdf['end_time_timestamp'])
    return newdf

def playAccountTable():
    newdf=pd.read_csv("data/account713.csv")
    # print("selected feauters:",newdf.columns)
    count=newdf['id'].value_counts()
    # if count.values.sum() == len(count):
    #     print("No duplicated data in the frame")
    # else:
    #     print("you need to go back for cleaning data ")
    newdf['nickname'].fillna(0,inplace=True)
    newdf['nickname']=newdf['nickname'].replace({'\\N': 0})
    newdf['nickname']=newdf['nickname'].apply(lambda x:1 if isinstance(x,str) else x)

    newdf['name'].fillna(0,inplace=True)
    newdf['name']=newdf['name'].replace({'\\N': 0})
    newdf['name']=newdf['name'].apply(lambda x:1 if isinstance(x,str) else x)

    newdf['sex'].fillna(0,inplace=True)
    newdf['sex']=newdf['sex'].replace({'\\N': 0})

    newdf['head_url'].fillna(0,inplace=True)
    newdf['head_url']=newdf['head_url'].replace({'\\N': 0})
    newdf['head_url']=newdf['head_url'].apply(lambda x:1 if isinstance(x,str) else x)

    # newdf['region']=newdf['region'].apply(lambda x:0 if not isinstance(x,str) else x)"数字也是字符串，不好处理"
    newdf['region'].fillna('0',inplace=True)
    newdf['region']=newdf['region'].replace({'\\N': '0'})
    newdf['region']=category_to_number(newdf.region,'region')
    # # newdf['region']=newdf['region'].apply(lambda x:1 if isinstance(x,str) else x)
    newdf['autograph'].fillna(0,inplace=True)
    newdf['autograph']=newdf['autograph'].replace({'\\N': 0})
    newdf['autograph']=newdf['autograph'].apply(lambda x:1 if isinstance(x,str) else x)

    newdf['integral'].fillna(0,inplace=True)
    newdf['integral']=newdf['integral'].replace({'\\N': 0})

    newdf['is_big_v'].fillna(0,inplace=True)
    newdf['is_big_v']=newdf['is_big_v'].replace({'\\N': 0})

    newdf['is_first_login'].fillna(0,inplace=True)
    newdf['is_first_login']=newdf['is_first_login'].replace({'\\N': 0})
    newdf['is_first_login']=newdf['is_first_login'].astype('int')

    # print(newdf.isnull().any()) #判断那一列有空
    newdf.loc[:,['creator_id','news_title','news_subtitle',
                            'updated_year','updated_month','updated_day',
                            'updated_timestamp','newsID']]
    df_toWrite=newdf.loc[:,['id','nickname','name','sex','head_url','region','autograph','integral','is_big_v','is_first_login']]
    df_toWrite.dropna(axis=0, how='any', inplace=True)
    df_toWrite.to_csv("data/clean_account.csv")


def playNewsTable():
    newdf=pd.read_csv("data/news713.csv",parse_dates=[2,4])
    # print("original features:",newdf.columns)
    count=newdf['id'].value_counts()
    # if count.values.sum() == len(count):
    #     print("No duplicated data in the frame， length=",len(count))
    # else:
    #     print("you need to go back for cleaning data ")
    # print(newdf.isnull().any()) #which column is null
    # print(newdf.isnull().any()) #which column is null
    newdf['creator_id'].fillna(0,inplace=True)

    # toReplace=newdf['updated_time'][newdf['updated_time'].isnull()].index.to_list()#replace null of updated_time with update_time
    # for i in toReplace:
    # #     print("oiginal:",newdf.ix[i,'updated_time'])
    #     newdf.ix[i,'updated_time']=newdf.ix[i,'updated_time']
    # #     print("filled:",newdf.ix[i,'updated_time'])
    dates = pd.to_datetime(newdf['updated_time'], format="%Y-%m-%d %H:%M:%S")
    # dates = pd.to_datetime(newdf['publish_time'], format="%Y-%m-%d %H:%M:%S") #some problem on converting str to datetime
    # print(type(dates))

    newdf['updated_year']=dates.dt.year
    newdf['updated_month']=dates.dt.month
    newdf['updated_week']=dates.dt.week
    newdf['updated_day']=dates.dt.day

    newdf['updated_time']=newdf['updated_time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    newdf['updated_timestamp'] = newdf['updated_time'].apply(lambda x:time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))
    def minMaxNormalization(x):
        return DataFrame({"updated_timestamp":np.rint([1000*(float(i)-min(x))/float(max(x)-min(x)) for i in x])})
    newdf['updated_timestamp']=minMaxNormalization(newdf['updated_timestamp'])
    newdf['news_title']=newdf['news_title'].apply(lambda x:1 if isinstance(x,str) else 0)
    newdf['news_subtitle']=newdf['news_subtitle'].apply(lambda x:1 if isinstance(x,str) else 0)
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


    newdf['newsID']=newdf['id']
    df_toWrite=newdf.loc[:,['creator_id','news_title','news_subtitle',
                            'updated_year','updated_month','updated_day',
                            'updated_timestamp','newsID']]
    df_toWrite.dropna(axis=0, how='any', inplace=True)
    df_toWrite.to_csv("data/clean_news.csv")

def playTopicNewsTable():
    newdf=pd.read_csv("data/topic713.csv",parse_dates=[10,16,17])

    # print("original features:",newdf.columns)
    # print(newdf.isnull().any()) #which column is null
    newdf=decode_time(newdf,'start_time')
    newdf=decode_time(newdf,'end_time')

    def check_deletedFlag():
        flags =newdf['delete_flag'].tolist()
        removeIndexs=[]
        for index, flag in enumerate(flags):
            if flag == 1:
                removeIndexs.append(index)
        newdf.drop(newdf.index[removeIndexs],inplace=True)
        # print(removeIndexs[:20])

    def check_publishedFlag():
        flags =newdf['publish_flag'].tolist()
        removeIndexs=[]
        for index, flag in enumerate(flags):
            if flag == 0:
                removeIndexs.append(index)
        newdf.drop(newdf.index[removeIndexs],inplace=True)
        # print("remove unpublished id:(show 20)",removeIndexs[:20])
    check_publishedFlag()
    check_deletedFlag()
    # print("droped: ",newdf['end_time_timestamp'])

    newdf['status']=category_to_number(newdf.status.values,'status')
    newdf['topicID']=newdf['id']
    df_toWrite=newdf.loc[:,['creator_id','discuss_num','sort_no','topic_type'
                    ,'start_time_year','start_time_month','start_time_day','start_time_week','start_time_timestamp'
                    ,'end_time_year','end_time_month','end_time_day','end_time_week','end_time_timestamp','liked_num'
                    ,'clap_num','top_flag','vote_num','status'
                    ,'hot_flag','topicID']]
    df_toWrite.dropna(axis=0, how='any', inplace=True)
    df_toWrite.to_csv("data/clean_topic.csv")

def playViewedNewsTable():
    newdf=pd.read_csv("data/click713.csv")
    # print("orginal feauters:",newdf.columns,' len:',len(newdf),' real user(viewing articles):',len(newdf.user_id.unique()))
    # print(newdf.isnull().any()) #which column is null
    newdf.drop(newdf['article_id'][newdf['article_id'].isnull()].index.to_list(), inplace=True)
    # toReplace1=newdf['end_time'][newdf['end_time'].isnull()].index.to_list()

    def remove_useless():
        flags =newdf['user_id'].tolist()
        removeIndexs=[]
        for index, flag in enumerate(flags):
            if flag == 'unlogin':
                removeIndexs.append(index)

        flags1 =newdf['article_id'].tolist()
        for index1, flag1 in enumerate(flags1):
            if flag1 == '\\N':
                removeIndexs.append(index1)
        newdf.drop(newdf.index[removeIndexs],inplace=True)

        # print(removeIndexs[:20],len(newdf.user_id.unique()))
    remove_useless()
    df_toWrite=newdf.loc[:,['user_id','article_id']]
    df_toWrite.dropna(axis=0, how='any', inplace=True)
    df_toWrite.to_csv("data/clean_click.csv")

def buildTrainingData():

    dfnews=pd.read_csv('data/clean_news.csv')
    dfuser=pd.read_csv('data/clean_account.csv')
    dfclick=pd.read_csv('data/clean_click.csv')

    dfnews.rename(columns={'newsID':'newsid'}, inplace = True)
    dfclick.rename(columns={'user_id':'userid','article_id':'newsid'}, inplace = True)
    dfuser.rename(columns={'id':'userid'}, inplace = True)
    dfnews.drop(dfnews.columns[0], axis=1, inplace=True)       # 删除第1列
    dfclick.drop(dfclick.columns[0], axis=1, inplace=True)       # 删除第1列
    dfuser.drop(dfuser.columns[0], axis=1, inplace=True)       # 删除第1列

    print("clicktbale  ",dfclick.columns, "table len=",len(dfclick),"user number=",len(dfclick.userid.unique()))
    print('----------'*10)
    print("usertable  ",dfuser.columns, "table len=",len(dfuser),"user number=",len(dfuser.userid.unique()))
    print('----------'*10)
    print("newtable  ",dfnews.columns, "table len=",len(dfnews),"news number=",len(dfnews.newsid.unique()))
    print('----------'*10)



    pos_merge1=pd.merge(dfclick,dfuser,on='userid',how='inner')
    print("merged1 ",pos_merge1.columns, "table len=",len(pos_merge1),"userid number=",len(pos_merge1.userid.unique()))
    print('----------'*10)
    pos_merge2=pd.merge(pos_merge1,dfnews,on='newsid',how='inner')
    print("merged2 ",pos_merge2.columns, "table len=",len(pos_merge2),"userid number=",len(pos_merge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    pos_merge2['label']=1
    pos_merge2.dropna(axis=0, how='any', inplace=True)
    pos_merge2.to_csv('ods_sql_postive_data.csv')
    length=len(pos_merge2)



    negative_articles=set(dfnews['newsid'].to_list())^set(dfclick['newsid'].tolist())
    # negative_users=set(dfuser['userid'].to_list)^set(dfclick['userid'].tolist())
    negative_clickdf=pd.DataFrame({'userid':dfclick['userid'].to_list()[:length],'newsid':list(negative_articles)[:length]})
    negmerge1=pd.merge(negative_clickdf,dfuser,on='userid',how='inner')
    print("merged1 ",negmerge1.columns, "table len=",len(negmerge1),"userid number=",len(negmerge1.userid.unique()))
    print('----------'*10)
    negmerge2=pd.merge(negmerge1,dfnews,on='newsid',how='inner')
    print("merged2 ",negmerge2.columns, "table len=",len(negmerge2),"userid number=",len(negmerge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    negmerge2['label']=0
    negmerge2.dropna(axis=0, how='any', inplace=True)
    negmerge2.to_csv('negative_data.csv')

    negative_articles=set(dfnews['newsid'].to_list())^set(dfclick['newsid'].tolist())
    # negative_users=set(dfuser['userid'].to_list)^set(dfclick['userid'].tolist())
    negative_clickdf=pd.DataFrame({'userid':dfclick['userid'].to_list()[:length],'newsid':list(negative_articles)[:length]})
    negmerge1=pd.merge(negative_clickdf,dfuser,on='userid',how='inner')
    print("merged1 ",negmerge1.columns, "table len=",len(negmerge1),"userid number=",len(negmerge1.userid.unique()))
    print('----------'*10)
    negmerge2=pd.merge(negmerge1,dfnews,on='newsid',how='inner')
    print("merged2 ",negmerge2.columns, "table len=",len(negmerge2),"userid number=",len(negmerge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    negmerge2['label']=0
    negmerge2.dropna(axis=0, how='any', inplace=True)
    negmerge2.to_csv('negative_data.csv')


    trainingdf=shuffle(pd.concat([negmerge2,merge2]))
    # print("columns0 ",trainingdf.columns)
    trainingdf.reset_index()
    # print("columns1 ",trainingdf.columns)
    # trainingdf.drop('index', axis=1, inplace=True)       # 删除第1列
    # print(trainingdf.columns)
    trainingdf.to_csv('traningset.csv')


    merge2['useridReindex']=category_to_number_onNewColumn(merge2.userid,'userid')
    merge2['newsidReindex']=category_to_number_onNewColumn(merge2.newsid,'newsid')
    merge2.to_csv('lfm_postive_data.csv')

    merge1=pd.merge(dfclick,dfuser,on='userid',how='inner')
    print("merged1 ",merge1.columns, "table len=",len(merge1),"userid number=",len(merge1.userid.unique()))
    print('----------'*10)
    merge2=pd.merge(merge1,dfnews,on='newsid',how='inner')
    print("merged2 ",merge2.columns, "table len=",len(merge2),"userid number=",len(merge2.userid.unique()))
    # merge2.drop(merge2.columns[0], axis=1, inplace=True)       # 删除第1列
    merge2['label']=1
    merge2.dropna(axis=0, how='any', inplace=True)
    merge2.to_csv('postive_data.csv')
    length=len(merge2)

    trainingdf=shuffle(pd.concat([negmerge2,merge2]))
    # print("columns0 ",trainingdf.columns)
    trainingdf.reset_index()
    # print("columns1 ",trainingdf.columns)
    # trainingdf.drop('index', axis=1, inplace=True)       # 删除第1列
    # print(trainingdf.columns)
    trainingdf.to_csv('traningset.csv')


    merge2['useridReindex']=category_to_number_onNewColumn(merge2.userid,'userid')
    merge2['newsidReindex']=category_to_number_onNewColumn(merge2.newsid,'newsid')
    merge2.to_csv('lfm_postive_data.csv')




if __name__ == '__main__':
    cleanData()
