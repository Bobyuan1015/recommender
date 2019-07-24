#-*- coding: utf-8 -*-
'''
Created on 2019-07-13
To support data processing: data cleaning and feature building
@author: yuanfang
'''

import numpy as np
from pandas import DataFrame
import datetime
import datetime as dt
import linecache
import random
import math
import time
import os
import pandas as pd

random.seed(0)

class Utils(object):

    def excel_to_csv(self,file,to_file):
        data_xls=pd.read_excel(file,sheet_name=0)
        data_xls.to_csv(to_file,encoding='utf_8_sig')

    def read_path(self,path):
        dirs=os.listdir(path)
        return dirs

    def category_to_number(self,x,key):
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

    def birthday_to_age(self,x):
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

    def category_to_number_onNewColumn(self,x,column):
            listedArray = list(x)
            dic={}
            for index,item in enumerate(np.unique(x)):
                dic[item]=index
            for st_index, i in  enumerate(listedArray):
                listedArray[st_index]=dic[i]
            return DataFrame({column+'Reindex':listedArray})

    def datetime_toString(self,dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    def string_toDatetime(self,st):
        return datetime.datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    def string_toTimestamp(self,st):
        return time.mktime(time.strptime(st, "%Y-%m-%d %H:%M:%S"))

    def decode_time(self,newdf,column):
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
                    newdf['start_time'][index]=self.datetime_toString(newdf['start_time'][index])#.strftime("%Y-%m-%d %H:%M:%S")
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
                    newdf['end_time'][index1]=self.string_toDatetime(newdf['end_time'][index1])
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

    def decompose_time(self,name,newdf):
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
                newdf[name][index1]=self.string_toDatetime(newdf[name][index1])
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
        newdf[name+'_timestamp'] = newdf[name].apply(lambda x:self.string_toTimestamp(self.datetime_toString(x)))
        print("decompose time  year month day week cost",time.time()-previous_time)
        previous_time= time.time()
        # print("replaced: ",newdf['end_time_timestamp'])
        def minMaxNormalization(x):
            return DataFrame({"updated_timestamp":np.rint([1000*(float(i)-min(x))/float(max(x)-min(x)) for i in x])})
        newdf[name+'_timestamp']=minMaxNormalization(newdf[name+'_timestamp'])
        print("decompose time  minMaxNormalization cost",time.time()-previous_time)

        return newdf

    def justSaveNewsDetail(self):
        print('just save news content into a file')
        df=pd.read_csv('data/new-temp_user_news.csv',index_col=0)
        print(df.columns)
        print(df['news_detail'][df['news_detail'].isnull()].index.to_list())
        df_toWrite=df.loc[:,['news_id','news_title', 'news_subtitle', 'news_detail']]
        print(df_toWrite.columns)
        df_toWrite.to_csv('news.csv')
        df=pd.read_csv('news.csv',index_col=0)
        print(df.columns)

    def removeTapIn_oneColumn(self,filename):
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
