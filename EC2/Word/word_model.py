import boto3
import requests
import json
import datetime
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
from hanspell import spell_checker
import nltk
from konlpy.tag import Okt
from tqdm import tqdm

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from keras.utils import pad_sequences

DATE = int(str(datetime.datetime.now() - datetime.timedelta(1))[:10].replace('-', ''))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cpu()
model.load_state_dict(torch.load('/home/ubuntu/Word/model_crawling_bert.pt', map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

okt = Okt()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = open('/home/ubuntu/Word/korean_stopwords.txt').read()
stop_words = stop_words.split('\n')


def upload_file_s3(bucket, file, content):
    try:
        csv_buffer = StringIO()
        content.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, file).put(Body=csv_buffer.getvalue())
        return True
    except:
        return False


def ls_s3(bucket, folder):
    resource = boto3.resource('s3')
    try:
        for obj in resource.Bucket(bucket).objects.filter(Prefix=f'{folder}/'):
            print(obj.key)
        return True
    except:
        return False


def download_file_s3(bucket, folder, file, path='./'):
    client = boto3.client('s3')
    file = client.get_object(Bucket=bucket, Key=f'{folder}/{file}')
    body = file['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df


def get_top10_movies():
    url = f'http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=5d450765c401dac01475905006e44d99&targetDt={DATE}'
    res = requests.get(url)
    txt = res.text
    result = json.loads(txt)

    movie_titles = []
    for movie_info in result['boxOfficeResult']['dailyBoxOfficeList']:
        movie_titles.append(movie_info['movieNm'])
    
    review_urls = []
    movie_codes = []
    for movie_title in movie_titles:
        review_url, movie_code = get_review_url(movie_title)
        review_urls.append(review_url)
        movie_codes.append(movie_code)
    
    return movie_titles, review_urls, movie_codes


def get_review_url(movie_title):
    url = f'https://movie.naver.com/movie/search/result.naver?query={movie_title}&section=all&ie=utf8'
    res = requests.get(url)
    
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        user_dic = {}
        idx = 1
        for href in soup.find("ul", class_="search_list_1").find_all("li"):
            user_dic[idx] = int(href.dl.dt.a['href'][30:])
            idx += 1
        
        movie_code = user_dic[1]
        order_key = 'newest'  # 공감순: order=sympathyScore, 최신순: order=newest
        url = f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={movie_code}&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order={order_key}&page='
        return url, movie_code
    else:
        print(f'Request Error: Code {res.status_code}')
        return '', 0


def get_img_src(movie_code):
    url = f'https://movie.naver.com/movie/bi/mi/basic.naver?code={movie_code}'
    res = requests.get(url)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        img_src = soup.select_one('#content > div.article > div.mv_info_area > div.poster > a > img').attrs['src']
        return img_src
    else:
        print(f'Request Error: Code {res.status_code}')
        return ''


def convert_input_data(sentences):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    return inputs, masks


def test_sentences(sentences):
    model.eval()
    inputs, masks = convert_input_data(sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    return logits


def analysis(df):
    # 각 리뷰에 대해 긍부정 라벨링
    df['Label'] = np.nan
    for i in tqdm(range(len(df))):
        try:
            logits = test_sentences([df.loc[i, '리뷰']])
            df.loc[i, 'Label'] = np.argmax(logits)
        except:
            pass
    
    return df


def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''} 

    for s in specials:
        text = text.replace(s, specials[s])
   
    return text.strip()


def clean_text(texts):
    corpus = []
    for i in range(len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
        review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        review = re.sub(r'♥','',review)
        review = re.sub(r'~','',review)
        review = re.sub(r'&','',review)
        corpus.append(review)
    
    return corpus


def grammar_check(text):
    spelled_sent = spell_checker.check(text)
    hanspell_sent = spelled_sent.checked
    return hanspell_sent


def tokenize_tagged(text):
    temp_X = okt.morphs(text, norm=True, stem=False) # 토큰화
    temp_X = [word for word in temp_X if word not in stop_words] # 불용어 제거
    return " ".join(temp_X)


def preprocess(df):
    df['전처리완료'] = ''
    for i in tqdm(range(len(df))):
        if type(df.loc[i, '리뷰']) == str:
            text = df.loc[i, '리뷰']
            text1 = clean_punc(text, punct, punct_mapping)
            text2 = ''.join(clean_text(text1))
            text3 = grammar_check(text2)
            text4 = tokenize_tagged(text3)
            df.loc[i, '전처리완료'] = text4
    
    return df


MOVIE_TITLES, REVIEW_URLS, MOVIE_CODES = get_top10_movies()
IMAGE_SRCS = [get_img_src(movie_code) for movie_code in MOVIE_CODES]
review_df = download_file_s3('movietomodel', 'word', f'word-{DATE}.csv').drop('Unnamed: 0', axis=1)
review_df = analysis(review_df)
review_df = preprocess(review_df)
review_df['사진'] = review_df['제목'].apply(lambda x: IMAGE_SRCS[MOVIE_TITLES.index(x)])
upload_file_s3('modeltoelastic', f'word/word-{DATE}.csv', review_df)