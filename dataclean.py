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
        dic={}
        for index,item in enumerate(np.unique(x)):
            dic[item]=index
        for st_index, i in  enumerate(listedArray):
            listedArray[st_index]=dic[i]
        return DataFrame({key:listedArray})


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


def removeTapIn_oneColumn():
    fp = open('data/news_title_comment_like.txt', 'r')
    fp1 = open('data/corrct_news_title_comment_like.txt', 'wb')
    correctLines=[]
    tmp=['']
    for i, line in enumerate(fp):
            num = (len(line) - len(line.replace(':!:',""))) // len(':!:')
            if num == 10 :
                print(' write:')
                fp1.write(bytes(line,'UTF-8'))
            else:
                tmp[len(tmp)-1]+=line.strip('\r\n')
                tempnum = (len(tmp[len(tmp)-1]) - len(tmp[len(tmp)-1].replace(':!:',""))) // len(':!:')
                if tempnum ==10:
                    fp1.write(bytes(tmp[len(tmp)-1],'UTF-8'))
                    tmp.append('')
                    print("append tempnum=",tempnum,'----len of tmp=',len(tmp[len(tmp)-1]),' index=',len(tmp)-1,' ',tmp[len(tmp)-1])
    print("converting job is done, you can check the file.")


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
    newdf['region']=category_to_number(newdf.region.values,'region')
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
        print("remove news:",removeIndexs[:-1])
        newdf.drop(newdf.index[removeIndexs],inplace=True)

    removeItems()


    newdf['newsID']=newdf['id']
    df_toWrite=newdf.loc[:,['creator_id','news_title','news_subtitle','updated_year','updated_month','updated_day','updated_timestamp','newsID']]
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

    merge2['useridReindex']=category_to_number_onNewColumn(merge2.userid,'userid')
    merge2['newsidReindex']=category_to_number_onNewColumn(merge2.newsid,'newsid')
    merge2.to_csv('lfm_postive_data.csv')


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


def main():
    pd.set_option('display.width',None)#dataframe can print full columns

    # playAccountTable()
    # playNewsTable()
    # playTopicNewsTable()
    # playViewedNewsTable()
    buildTrainingData()

if __name__ == '__main__':
    main()
