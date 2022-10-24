#!/usr/bin/env python
# coding: utf-8

# # 모델에 들어가지 않는 데이터

# 영화 상세정보 api에 크롤링 2개 merge한 데이터 프레임명이 `df_info` 일때

# ## 1) 장르, 등급, 감독, 국적, 배급사 값 추출

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import boto3
import datetime
from io import StringIO


date = str(datetime.datetime.now() - datetime.timedelta(1))[:10].replace('-', '')


def download_file_s3(bucket):
    client = boto3.client('s3')
    bucket = 'movietomodel'
    file = client.get_object(Bucket = bucket, Key=f'yuha/yuha-{date}.csv')
    body = file['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df



df_info = download_file_s3('modeltomovie')
df_info = df_info.drop('Unnamed: 0', axis = 1)


# ## 3) 배급사/감독 흥행지수 조인


distri_index = pd.read_csv('/home/ubuntu//Predict/company.csv') #저장된 경로로 바꾸기
direc_index = pd.read_csv('/home/ubuntu//Predict/director.csv')

mean_distri = distri_index.mean()[0]
mean_direc = direc_index.mean()[0]


def add_index(data):
  data = data.merge(distri_index[['배급사', '흥행지수1']],on='배급사', how='left')
  data = data.merge(direc_index[['감독', '흥행지수1']],on='감독', how='left')
  data.rename(columns={'흥행지수1_x':'배급사 흥행지수', '흥행지수1_y':'감독 흥행지수'}, inplace=True)
  data['배급사 흥행지수'] = data['배급사 흥행지수'].fillna(mean_distri)
  data['감독 흥행지수'] = data['감독 흥행지수'].fillna(mean_direc)
  return data



data = add_index(df_info)


# # 모델 전처리~ 로드

# ## 1) 모델에 들어갈 데이터 전처리


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pickle
import joblib


# 모델에 넣을 용도 ; 필요한 컬럼만 가져오기 (맨마지막 실행)
def drop_columns(df):
  test = df[['movieCd','openDt','감독 흥행지수', '뉴스 언급 건수','배급사 흥행지수', '장르', '국적', '등급', '평점']]
  test['movieCd'] = df['movieCd'].astype(int)
  test['openDt'] = df['movieCd'].astype(int)
  return test


test = drop_columns(data)


# ## 2) 라벨 인코딩


def change_genre(x):
  genre_list = ['드라마', '판타지', 'SF', '액션', '애니메이션', '다큐멘터리', '사극', '코미디', '어드벤처','스릴러', '범죄', '공포(호러)', '멜로/로맨스', '뮤지컬', '미스터리', '공연', '가족', '전쟁','기타', '서부극(웨스턴)']
  return genre_list.index(x)

def change_nation(x):
  if x == '한국':
    return 2
  elif x == '미국':
    return 1
  else:       # '한국미국제외'
    return 0 

def change_grade(x): 
  if x == '전체관람가':
    return 3
  elif x == '12세이상관람가':
    return 2
  elif x == '15세이상관람가':
    return 1
  else:
    return 0 


test['장르'] = test['장르'].map(change_genre) 
test['국적'] = test['국적'].map(change_nation) 
test['등급'] = test['등급'].map(change_grade) 


# ## 3) 모델 load


final_model = joblib.load('/home/ubuntu/Predict/modelpredict_final_model_label.pkl')  # path 추가 필요!

# ## 4) 예측


test_features = test.drop(["movieCd"], axis = 1)

test_predict = final_model.predict(test_features)


test['predict'] = test_predict


data['predict'] = test['predict']
data


# # 3.일별 박스오피스 데이터에서 대시보드에 들어가야 하는 데이터


import json
import pandas as pd
import numpy as np
import requests
import re


def getMovie10() :
    url = f'http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=5d450765c401dac01475905006e44d99&targetDt={date}'
    res = requests.get(url)
    text= res.text
    movie_raw = json.loads(text)
    df = pd.DataFrame()
    l1 = []; l2 = []; l3 = []; l4 = []; l5 =[]; l6 = []; l7 = [];
    l8 = []; l9 = []; l10 = []; l11 = []; l12 = []; l13 = []; l14 = []; l15 = [];
    
    for movie_info in movie_raw['boxOfficeResult']['dailyBoxOfficeList']:
        l1.append(movie_info['movieNm'])
        l2.append(movie_info['openDt'])
        l3.append(movie_info['movieCd']) 
        l4.append(movie_info['rank'])
        l5.append(movie_info['rankInten'])
        l6.append(movie_info['rankOldAndNew'])
        l7.append(movie_info['salesAmt'])
        l8.append(movie_info['salesShare'])
        l9.append(movie_info['salesChange'])
        l10.append(movie_info['salesAcc'])
        l11.append(movie_info['audiCnt'])
        l12.append(movie_info['audiInten'])
        l13.append(movie_info['audiChange'])
        l14.append(movie_info['audiAcc'])
        l15.append(movie_info['audiChange'])
    
    df['movieNm'] = l1
    df['openDt'] = [int(re.sub('-','',i)) for i in l2]
    df['movieCd'] = [int(i) for i in l3 ]  
    df['rank'] = [int(i) for i in l4]   
    df['rankInten'] = [int(i) for i in l5]   
    df['rankOldAndNew'] = l6   
    df['salesAmt'] = [int(i) for i in l7]   
    df['salesShare'] = [float(i) for i in l8]   
    df['salesChange'] = [float(i) for i in l9]   
    df['salesAcc'] = [int(i) for i in l10]   
    df['audiCnt'] = [int(i) for i in l11]   
    df['audiInten'] = [int(i) for i in l12]   
    df['audiChange'] = [float(i) for i in l13]     
    df['audiAcc'] = [float(i) for i in l14]     
    df['audiChange'] = [float(i) for i in l15]      
    return df


data2 = getMovie10()

data2['movieCd'] = data2['movieCd'].astype(int)
data_all = pd.merge(data, data2, how='left', on = ['movieCd', 'openDt','movieNm'])

data_last = data_all.drop(['movieNmEn', 'movieNmOg', 'prdtYear', 'nations', 'genres', 'directors', 'actors', 'showTypes', 'companys', 'audits', 'staffs', 'openDt_c', 'startDt', 'endDt'], axis=1)

data_last = data_last.to_csv()


def upload_file_s3(bucket,filename, content):
    client = boto3.client('s3')
    try :
        client.put_object(Bucket=bucket, Key=filename, Body=content)
        return True
    except :
        return False


bucket = 'modeltoelastic'
upload_file_s3(bucket, f'yuha/yuha-{date}.csv', data_last)

print('end')