import boto3
import json
from requests_aws4auth import AWS4Auth
import requests
import datetime
import pandas as pd
from io import StringIO

date = str(datetime.datetime.now() - datetime.timedelta(1))[:10].replace('-', '')
date_type = str(datetime.datetime.now() - datetime.timedelta(1))[:10]

def lambda_handler(event, context):
    host = 'https://search-conference-movie-dashboard-ndmh3o6ttgocn6g27z7zjkcvia.ap-northeast-2.es.amazonaws.com'
    region = 'ap-northeast-2'
    service = 'es'
    access_key = 'AKIA43WI62CSMLMM4L'
    secret_key = '2yPxXDNJbV3drPNaK7ID7qJxeUYdaDhYu7xqaM'
    awsauth = AWS4Auth(access_key, secret_key, region, service)
    headers = {"Content-Type": "application/json"}
    df_yuha = download_file_s3_yuha()
    df_yuha_json = to_json_yuha(df_yuha)
    _index = 'yuha'
    for _id, data in enumerate(df_yuha_json):
        _id = date+'-'+str(_id)  
        url = f'{host}/{_index}/_doc/{_id}'
        data = json.loads('{' + data + '}')
        y = requests.post(url, json=data, headers=headers, auth=awsauth)
    print(y)
    
    df_word = download_file_s3_word()
    df_word_json = to_json_word(df_word)
    _index = 'word'
    for _id, data in enumerate(df_word_json):
        _id = date+'-'+str(_id)    
        url = f'{host}/{_index}/_doc/{_id}'
        data = json.loads('{' + data + '}')
        w = requests.post(url, json=data, headers=headers, auth=awsauth)
    print(w)    
    return {
        'statusCode': 200,
        'body': print('Hello from Lambda!')
    }


def download_file_s3_yuha():
    client = boto3.client('s3')
    bucket = 'modeltoelastic'
    file = client.get_object(Bucket = bucket, Key='yuha/yuha-'+date+'.csv')
    body = file['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    df['crawlDate'] = date_type 
    df['openDate'] = df['openDt'].apply(lambda x : str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
    df = df.drop('openDt', axis=1)
    df = df.rename(columns={'movieNm':'제목'})
    return df


def download_file_s3_word():
    client = boto3.client('s3')
    bucket = 'modeltoelastic'
    file = client.get_object(Bucket = bucket, Key='word/word-'+date+'.csv')
    body = file['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    df['crawlDate'] = date_type 
    return df
    
    
# 댓글 및 감성분석 및 이미지 소스 csv파일 -> 엘라스틱에 넣을 json파일
def to_json_word(df):
    for i in range(len(df)):
        for col in df.columns:
            if type(df.loc[i, col]) == str:
                df.loc[i, col] = df.loc[i, col].replace('"', "'")

    return df.to_json( orient='records', force_ascii=False)[2:-2].split('},{')


# 영화 정보 csv 파일 -> 엘라스틱에 넣을 json파일
def to_json_yuha(df):
    return df.to_json(orient='records', force_ascii=False)[2:-2].split('},{')