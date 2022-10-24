import json
import pandas as pd
import numpy as np
import requests
import boto3
import re
from datetime import datetime, timedelta 
from bs4 import BeautifulSoup
from itertools import chain

def lambda_handler(event, context):
    bucket = 'movietomodel'
    date = str(datetime.now() - timedelta(1))[:10].replace('-', '')
    file_name = 'yuha-'+date
    df= getMovie10()
    df_info = movie_info(df)
    df_news = news_crawler(df_info)
    df_star = star_crawler(df_info)
    df_info = pd.merge(df_info, df_news, how='left', on = 'movieCd')
    df_info = pd.merge(df_info, df_star, how='left', on = 'movieCd')
    df_info = extract_value(df_info)
    df_info = pre_nation(df_info)
    result = df_info.to_csv()
    result = upload_file_s3(bucket,'yuha/' + file_name + '.csv', result)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
    
    if result:
        return {
            'statusCode': 200,
            'body': json.dumps("upload success")
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps("upload fail")
        }


def date_format(date) : 
    date = str(date)[:10]
    date = date.replace('-','')
    return date

def getMovie10() :
    date = str(datetime.now() - timedelta(1))[:10].replace('-', '')
    url = 'http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=5d450765c401dac01475905006e44d99&targetDt='+date
    res = requests.get(url)
    text= res.text
    movie_raw = json.loads(text)
    df = pd.DataFrame()
    l1 = []; l2 = []; l3 = []
    
    for movie_info in movie_raw['boxOfficeResult']['dailyBoxOfficeList']:
        l1.append(movie_info['movieNm'])
        l2.append(movie_info['openDt'])
        l3.append(movie_info['movieCd']) 
    
    df['movieNm'] = l1
    df['openDt'] = l2
    df['movieCd'] = l3      
    return df
    
def movie_url(moviecd):
  return 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json?key=dddea6429a20616861ab4a73b4618dc3&movieCd=' + str(moviecd)

def movie_info(df):
  moviecd_list = df['movieCd']
  movie_jsons = (requests.get(movie_url(moviecd)).json() for moviecd in moviecd_list)
  movie_json = chain(movie_json['movieInfoResult']['movieInfo'] for movie_json in movie_jsons)
  df_info = pd.json_normalize(movie_json)
  return df_info
  
def news_crawler(df):
    news_list = []
    movieCd_list = df['movieCd'].to_list()
    movie_list = df['movieNm'].to_list()
    keyword_list = [re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',' ', movie) for movie in movie_list]
    
    df['openDt_c'] = pd.to_datetime(df['openDt'], format = '%Y%M%d')
    df['startDt'] = df['openDt_c']- timedelta(days=15)
    df['endDt'] = df['openDt_c'] + timedelta(days=15)
    opendt_list = [df['openDt'].astype('str')[i].replace('-','') for i in range(len(df))]
    startdate_list = [df['startDt'].astype('str')[i].split(' ')[0].replace('-','') + '000000' for i in range(len(df))]
    enddate_list = [df['endDt'].astype('str')[i].split(' ')[0].replace('-','') + '235959' for i in range(len(df))]

    for i in range(len(movieCd_list)):
        movieCd = str(movieCd_list[i])
        keyword = str(keyword_list[i])
        startdate = str(startdate_list[i])
        enddate = str(enddate_list[i])
        url = f"https://search.daum.net/search?nil_suggest=btn&w=news&DA=STC&q=영화+{keyword}&p=1&period=u&sd={startdate}&ed={enddate}"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text,"html.parser")
            count = soup.select_one('#resultCntArea').get_text()
            count = count.split('/ ')[1]
            count = count.replace('약 ','')
            count = count.replace('건','')
            count = count.replace(',','')
            news_list.append([movieCd, count])
        
        else:
            print(response.status_code)
            
    news_df = pd.DataFrame(news_list, columns = ["movieCd", "뉴스 언급 건수"])
    news_df['뉴스 언급 건수'] = news_df['뉴스 언급 건수'].astype(int)
    return news_df

def star_crawler(df):
    star_list = []
    
    movieCd_list = df['movieCd'].to_list()
    movie_list = df['movieNm'].to_list()
    keyword_list = [re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',' ', movie) for movie in movie_list]
    
    df['openDt_c'] = pd.to_datetime(df['openDt'], format = '%Y%M%d')
    df['startDt'] = df['openDt_c']- timedelta(days=15)
    df['endDt'] = df['openDt_c'] + timedelta(days=15)
    
    for i in range(len(df)):
        movieCd = str(movieCd_list[i])
        keyword = str(keyword_list[i])
        url = f"https://movie.naver.com/movie/search/result.naver?query={keyword}&section=all&ie=utf8"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text,"html.parser")
            star =  soup.select('#old_content > ul.search_list_1 > li > dl > dd.point > em.num')
            if(len(star)>0):
                star = star[0].text
                star_list.append([movieCd, star])
            else: 
                star_list.append([movieCd, 0])
                
        
        else:
            print(response.status_code)
            
    star_df = pd.DataFrame(star_list, columns = ["movieCd", "평점"])
    star_df['평점'] = star_df['평점'].astype(float)
    return star_df

def extract_value(trial):
  # 장르, 등급, 감독, 국적 추출
  genre = []
  audit = []
  director = []
  nation = []
  for i in range(len(trial)):
    try :
      genre.append(trial.genres[i][0]['genreNm'])
    except :
      genre.append('')
    try : 
      audit.append(trial.audits[i][0]['watchGradeNm'])
    except :
      audit.append('')
    try :
      director.append(trial.directors[i][0]['peopleNm'])
    except :
      director.append('')
    try:
      nation.append(trial.nations[i][0]['nationNm'])
    except :
      nation.append('') 

  trial['장르'] = genre
  trial['등급'] = audit
  trial['감독'] = director
  trial['국적'] = nation

  #배급사 가져오기
  distributor = []
  for i in range(len(trial)):
    row = trial['companys'][i]
    for j in range(len(row)):
      if row[j]['companyPartNm'] == '배급사':
        distributor.append(row[j]['companyNm'])
        break
  trial['배급사'] = distributor
  return(trial)
 

def pre_nation(df):
  df.loc[(df['국적'] != '한국')*(df['국적']!='미국'), '국적'] = '한국미국제외'
  return df


def upload_file_s3(bucket,filename, content):
    client = boto3.client('s3')
    try :
        client.put_object(Bucket=bucket, Key=filename, Body=content)
        return True
    except :
        return False
        