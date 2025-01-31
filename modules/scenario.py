import argparse
import itertools
import os
import random
import re
import requests
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.utils import (
    get_naverlinks, fetch_article_details, get_jinga,
    get_study, get_study_format, get_output_format,
    get_shorturl, check_date
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


prompt_templates = {

    "system" : "You are a helpful journalist assistant designed to output JSON.",
        
    "digital" : 
    """당신의 업무는 뉴스 기사를 분류하는 것입니다. 
    주어진 뉴스 기사를 아래 조건에 따라 디지털 관련 기사면 1, 아니면 0으로 분류하세요.
    답변은 "디지털"이라는 키 값으로 JSON 형태로 추출하세요. 
    "You are a helpful assistant designed to output JSON."

    디지털 관련 기사로 판단하려면 다음 기준 중 하나 이상을 만족해야 합니다:

    1. 금융 기술(FinTech)이나 관련 디지털 솔루션에 대해 다루고 있는가? (온라인 뱅킹, 모바일 결제 시스템, 디지털 자산 관리 등)
    2. 블록체인 또는 암호화폐(가상화폐)에 대해 언급하고 있는가? (비트코인, 이더리움, 디지털 토큰 등)
    3. 디지털 결제 방식이나 시스템에 대해 설명하고 있는가? (전자지갑, 비대면 결제, NFC 결제 등)
    4. 금융 데이터 분석이나 알고리즘 트레이딩과 관련된 내용을 포함하고 있는가? (빅데이터 분석, 인공지능을 활용한 투자 전략 등)
    5. 디지털 금융 규제 또는 정책에 대해 이야기하고 있는가? (디지털 금융 보안, 개인정보 보호법, 전자금융거래법 등)
    6. 디지털 플랫폼을 이용한 금융 서비스나 제품을 다루고 있는가? (로보어드바이저, P2P 대출 플랫폼, 크라우드펀딩 등)
    7. 디지털화가 전통 금융 산업에 미치는 영향에 대해 논의하고 있는가? (은행의 디지털 트랜스포메이션, 금융기관의 IT 인프라 개선 등)

    ### 뉴스 기사
    {{content}}

    ### 답변
    디지털 해당여부(1 또는 0):
    """,

    "sentiment" : """다음 기사를 보고 아래 정보를 "긍정도" 키 값으로 JSON 형태로 추출하세요.
     "You are a helpful assistant designed to output JSON."
        긍정도 점수를 1 ~ 100 사이 점수로 표현하세요
        긍정적인 기사는 80점 이상 , 중립적인 기사는 60점 이하 50점 이상, 부정적인 기사는 20점 이허로 표현하세요

        #
        기사내용 : {{content}}
        긍정도 :
        """,

    "importance" : """당신은 전문 기자입니다. 다음 기사를 보고 아래 정보를 "중요도" 키 값으로 JSON 형태로 추출하세요.
     "You are a helpful assistant designed to output JSON."
        기사의 중요도를 점수 1에서 2까지로 표현(중요할수록 점수가 올라갑나다)
        신한은행, 신한 관련된 기사는 중요도 2을 부여하세요.
        디지털 관련 내용이면 중요도 2를 부여하세요. 예를 들어 AI, 핀테크 키워드가 있으면 중요도 2를 부여하세요.
        KB국민, 국민은행, 하나은행, 하나, 우리은행, 우리, 카카오뱅크, 토스뱅크, 케이뱅크의 디지털 주제와 관련된 기사는 중요도 2를 부여하세요
        적금 출시, 예금 출시와 같은 신상품에 관련된 기사는 중요도 2를 부여하세요.
        위 조건들에 해당되지 않는 기사는 중요도 1를 부여하세요


        # 예시
        예시 기사 제목 : 신한금융, 저출산 극복 위한 대체인력지원사업에 100억원 출연
        중요도  : 2

        예시 기사 제목 : 소비자 2000명이 블라인드 테스트로 뽑은 최고의 트래블 카드는?
        중요도 : 2

        예시 기사 제목 : 토스뱅크, 신용보증기금 '이지원 대출' 출시
        중요도 : 2

        예시 기사 제목 : 또 규제리스크 직면한 카카오뱅크···사업 다각화 ‘절실’
        중요도 : 2

        예시 기사 제목 : (5대 은행 AI 전쟁)⑤농협은행, 빠른 도입에도 성과 미진…효율화 '숙제'
        중요도 : 2

        예시 기사 제목 : 카드수수료 내년 또 내려가나…카드사 "인하여력 더는 없다" 반발 ★★★
        중요도 : 2
        
        예시 기사 제목 : 간편송금 악용 보이스피싱, 28일부터 신속히 차단한다 ★★★
        중요도 : 2

        예시 기사 제목 : 국내은행 상반기 순이익 12조6000억원…전년比 11%↓
        중요도 : 2

        예시 기사 제목 : "망분리 개선 보안 대책 어떻게?" 금융당국, 全 금융업권 대상 설명회 개최
        중요도 : 2

        예시 기사 제목 : 핀다, 내집 대출한도 계산기 오픈…LTV·DTI 한번에 계산
        중요도 : 1 
        
        # 실제
        예시 기사 제목 : {{content}}
        중요도 : 
        """,
        
    "status"  : 
        """당신은 전문 기자 입니다. 다음 기사를 제목을 보고 관련 내용이 "정부정책 및 동향" 관련 내용인지 "금융회사" 관련 된 내용인지 분류하세요. 
    아래 예시를 참고해서 "구분"이라는 키 값으로 JSON 형태로 추출하세요. "You are a helpful assistant designed to output JSON."
        신한은행, KB국민, 국민은행, 하나은행, 하나, 우리은행, 우리, 카카오뱅크, 토스뱅크, 케이뱅크 관련 기사는  "금융회사"에 해당되며
       금융감독원, 금감원, 정부부처, 정책, 시중은행, 인뱅3사 관련 기사는 "정부정책 및 동향"에 해당 됩니다.
       
        # 예시
        예시 기사 제목 : BC카드, 한국 금융 최적화된 AI 무상 공개
        예시 분류 : 금융회사

        예시 기사 제목 :  티몬,위메프에..은행권 선정산대출 취급 중단
        예시 분류 : 금융회사

        예시 기사 제목 : 토스뱅크-하나카드 함께 신용카드 만든다. PLCC업무협약 체결 
        예시 분류 : 금융회사

        예시 기사 제목 :  우리은행, IT서비스관리 국제표준 인증 획득
        예시 분류 : 금융회사
    
        예시 기사 제목 : "망분리 개선 보안 대책 어떻게?" 금융당국, 全 금융업권 대상 설명회 개최
        예시 분류 : 정부정책 및 동향

        예시 기사 제목 : 금감원, 가상자산거래소 예치금 이자율 공통기준 마련 요구
        예시 분류 : 정부정책 및 동향

        예시 기사 제목 : 규제샌드박스 전용펀드 175억원 조성 '신시장창출'
        예시 분류 : 정부정책 및 동향
        
        예시 기사 제목 : 인터넷은행도 중,저신용자 대출 줄였다
        예시 분류 : 정부정책 및 동향

        예시 기사 제목 : 국내은행 상반기 순이익 12조6000억원…전년比 11%↓
        예시 분류 : 정부정책 및 동향

        # 실제
        기사 내용 : {{content}}
        구분 :
        """,

    "duplicate" : "",
    
    "summary" : 
    """당신의 업무는 뉴스 기사를 간략하게 요약하는 것입니다.
    요약문을 "요약문" 키 값으로 JSON 형태로 추출하세요.
    "You are a helpful assistant designed to output JSON."

    1. 다음 뉴스 기사를 읽고, 핵심 내용을 3줄 이내로 간단하게 요약해주세요.
    2. 요약문을 존댓말로 해주세요.

    ### 뉴스 기사
    {{content}}

    ### 답변
    요약문(존댓말로):
    """
    ,

}



class main_bot():
    def __init__(self, load_option, date="", custom_url=""):
        print("Initialized!")
        self.load_option = load_option
        #self.credential = credential
        self.credential = st.secrets
        self.key = self.credential['chatgpt']['api_key']
        self.client = OpenAI(api_key=self.key)
        self.custom_url=custom_url

        if not date :
            self.target_date = datetime.strptime( (datetime.today() +  timedelta(days=1)).strftime('%Y%m%d'), "%Y%m%d")
        else : 
            self.target_date = datetime.strptime(date, "%Y%m%d")

        # 0. Collect + short url
        if not self.load_option :
            self.news_df = self.collect()
        else :
            self.load()
            #### For test
            #self.news_df = self.news_df[:10] ########### Temp


    def run(self):
        progress_text = "디지털 기사 추출"
        my_bar = st.progress(0, text=progress_text)
        
        # 1. Check Digital
        self.news_df = self.check_digital()
        my_bar.progress(5, text="긍정도 분석") 
        
        # 2. Sentiment
        self.news_df['sentiment'] = self.get_sentiment()
        my_bar.progress(15, text="중요도 분석")     
           
        # 3. importance & sort
        self.get_importance()
        my_bar.progress(30, text="동향 분류")        

        # 4. Classify - Status/ Trend 
        self.news_df['class'] = self.get_status()
        my_bar.progress(45, text="중복 제거")        

        # 5. Remove Duplicate
        self.news_df = self.remove_duplicate()
        my_bar.progress(80, text="요약")        

        # 6. Summary
        self.news_df["summary"] = self.get_summary()

        # 7. Template
        my_bar.progress(99, text="최종출력")        
        # 7.1 digital study
        self.digital_study =  self.get_study()
        response = self.digitaldaily_format()
        self.digitaldaily_format_all() # Temp

    def collect(self):
        
        if self.custom_url :
            with open("./urls.txt", "r") as f :
                lines = f.readlines()
            lines = [line.replace('\n', '') for line in lines]
            urls = lines
        else :
            urls = get_naverlinks(url="https://news.naver.com/breakingnews/section/101/259?date={}".format((self.target_date -  timedelta(days=1)).strftime('%Y%m%d')))
        news_list = [fetch_article_details(url) for url in urls]
        news_df = pd.DataFrame(news_list)
        # Remove no contents articles
        news_df = news_df[news_df['content'].apply(lambda x : len(x) != 0)]
        news_df = news_df.reset_index(drop=True)
        news_df = self.quick_remove_duplicate(news_df)
        news_df["shorturl"] = news_df["url"].map(lambda x : get_shorturl(x,self.credential))
        return news_df 
    
    def load(self):
        # 8월 3일 발송
        # 8월 2일 전체

        data_date = (self.target_date  -  timedelta(days=1)).strftime('%Y%m%d')
        load_path = "./data/news_df_" + data_date  + ".csv"
        self.news_df = pd.read_csv(load_path)
        idxs = self.news_df['time'].map(lambda x : check_date(x, (self.target_date  -  timedelta(days=1)).strftime('%Y.%m.%d.')))
        self.news_df = self.news_df[idxs]
        self.news_df = self.news_df.dropna().reset_index(drop=True)
        # 빠른 중복제거를 위한 TFIDF
        self.news_df = self.quick_remove_duplicate(self.news_df)
    
    def quick_calculate_similarity(self, article1, article2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([article1, article2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    
    def quick_remove_duplicate(self, news_df):
        news_pair = list(itertools.combinations(range(len(news_df)), 2))
        news_duplicate = []
        for (i, j) in news_pair:
            cos_sim = self.quick_calculate_similarity(news_df.loc[i, 'content'], news_df.loc[j, 'content'])
            if cos_sim >= 0.1:
                news_duplicate.append(j)
        news_unique = news_df.drop(list(set(news_duplicate)))
        return news_unique.reset_index(drop = True)
        
    def check_digital(self):
        news_df = self.news_df
        
        system_prompt = prompt_templates["system"] 
        environment = jinja2.Environment()
        jinja_template = environment.from_string(prompt_templates["digital"])

        digital_yn = []

        for content in news_df['content'] :
            prompt = jinja_template.render(content=content)
            response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            )
            try : 
                digital_yn.append((eval(response.choices[0].message.content)["디지털"]))
            except :
                digital_yn.append(-1)
                pass
            
        print("Check digital")
        
        news_df['digital_yn'] = digital_yn
        digital_news_df = news_df[news_df['digital_yn'] == 1].reset_index(drop = True)

        return digital_news_df 

    def get_sentiment(self):
        self.news_df = self.news_df
        news_df = self.news_df

        system_prompt = prompt_templates["system"] 
        environment = jinja2.Environment()
        jinja_template = environment.from_string(prompt_templates["sentiment"])

        scores = []

        for content in  news_df['content'] :
            prompt = jinja_template.render(content=content)
            response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            )
            try : 
                scores.append((eval(response.choices[0].message.content)["긍정도"]))
            except :
                scores.append(-1)
                pass
        print("Sentiment test")
        return scores

    def get_importance(self):
        news_df = self.news_df
        print(news_df)
        system_prompt = prompt_templates["system"]
        environment = jinja2.Environment()
        jinja_template = environment.from_string(prompt_templates["importance"])

        scores = []

        for title in  news_df['title'] :
            prompt = jinja_template.render(content=title)
            response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            )
            try : 
                scores.append((eval(response.choices[0].message.content)["중요도"]))
            except :
                scores.append(1)
                pass
        self.news_df['importance'] = scores

        # Importance가 2인데 신한 들어가 있으면 3점으로 바꾸는 로직
        scores = []
        for record in self.news_df.to_dict('records') :
            title, score = record['title'], record['importance']
            if ("토스" in title) or ("KB" in title) or ("우리은행" in title) or ("하나은행" in title) or \
             ("금감원" in title) or ("금융위" in title) or ("이복현" in title):
                score = min(score + 1 , 2)
            if ("신한" in title) or ("신한은행" in title) :
                score = min(score + 1 , 3)
            
            
            scores.append(score)
        self.news_df['importance']=scores
        self.news_df.sort_values(by=['importance', 'sentiment'], axis=0, ascending=False, inplace=True)
        print("importance test")
        return scores

    def get_status(self):
        news_df = self.news_df
        print(news_df)
        system_prompt = prompt_templates["system"]
        environment = jinja2.Environment()
        jinja_template = environment.from_string(prompt_templates["status"])

        scores = []

        for content in  news_df['title'] :
            prompt = jinja_template.render(content=content)
            response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
            )
            try : 
                scores.append((eval(response.choices[0].message.content)["구분"]))
            except :
                scores.append("null")
                pass
        print("status test")
        return scores
    
    def get_embedding_db(self):
        news_df = self.news_df 
        embedding_array = []
        for idx in range(len(news_df)):
            article = news_df.loc[idx, 'content']
            embedding = np.array(self.get_embedding(article)).reshape(1, -1)
            embedding_array.append(embedding)
        embedding_array = np.vstack(embedding_array)
        self.embedding_array = embedding_array
        

    def get_embedding(self, text):
        response = self.client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return response.data[0].embedding

    def calculate_similarity(self, article1, article2):
        embedding1 = self.embedding_array[article1].reshape(1, -1)
        embedding2 = self.embedding_array[article2].reshape(1, -1)
        cosine_sim = cosine_similarity(embedding1, embedding2)
        return cosine_sim[0][0]
    
    def remove_duplicate(self, threshold = 0.9):
        self.get_embedding_db()
        news_df = self.news_df
        news_pair = list(itertools.combinations(range(len(news_df)), 2))
        news_duplicate = []

        for (i, j) in news_pair:
            cos_sim = self.calculate_similarity(i, j)
            
            if cos_sim >= threshold:
                news_duplicate.append(j)
        
        news_unique = news_df.drop(list(set(news_duplicate)))
        return news_unique.reset_index(drop = True)

    def get_summary(self):
        news_df = self.news_df
        
        system_prompt = prompt_templates["system"]
        environment = jinja2.Environment()
        jinja_template = environment.from_string(prompt_templates["summary"])

        summary = []

        for i in range(len(news_df)) :
            prompt = jinja_template.render(content = news_df.loc[i, 'content'])
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
            )
            try : 
                summary.append((eval(response.choices[0].message.content)["요약문"]))
            except :
                summary.append("null")
                pass
                
        return summary
    
    def get_study(self):
        articles = get_study()
        self.study = get_study_format(articles, self.credential)


    def digitaldaily_format(self):
        # Key value check
        for key in ["publisher", "url"] :
            if key not in self.news_df.columns:
                self.news_df[key] = 'temp'

        status = get_output_format(self.news_df, status="금융회사")
        trend = get_output_format(self.news_df, status="정부정책 및 동향")
        study = self.study

        ####### To Do
        response = get_jinga(status, trend, study, target_date=self.target_date)
        print(response)
        with open("./data/result.txt", "a") as f:
            f.write(response)
        with st.chat_message("assistant"):
            st.code(response)

        return response
    def digitaldaily_format_all(self):
        # temp for recording
        for key in ["publisher", "url"] :
            if key not in self.news_df.columns:
                self.news_df[key] = 'temp'

        status = get_output_format(self.news_df, 
                                   status="금융회사", max_num=-1)
        trend = get_output_format(self.news_df, 
                                  status="정부정책 및 동향", max_num=-1)
        study = self.study

        ####### To Do
        # Save
        response = get_jinga(status, trend, study, target_date=self.target_date)
        with open("./data/result_all.txt", "w") as f:
            f.write(response)
        self.news_df.to_csv("news_df_{}".format(datetime.today().strftime('%Y%m%d')),index=False)

        expander = st.expander(label = "Advanced tools 🛠️")
        with expander :
            cols= st.columns((1,1))
            cols[0].download_button("전체 기사 다운로드", response, file_name="전체기사.txt")
            cols[1].button("새로고침")
            

        return response

# Parsing bot
class parse_bot():
    def __init__(self, save_date='', save_path="../data/news_df_") :
        self.credential = st.secrets
        self.key = credential['chatgpt']['api_key']
        self.fpath = save_path
        self.isexist_flag = False
        if save_date == '' :
            self.save_date = datetime.today().strftime('%Y%m%d')
        else :
            self.save_date = save_date
        # 1. Collect + short url
        self.collect_keywords = ["정상혁", "진옥동", "은행장", 
        "신한은행", "신한", "KB국민", "국민은행", "하나은행", "하나", "우리은행", "우리", "금융지주", 
         "카카오뱅크", "토스뱅크", "케이뱅크", "토스", "네이버", "카카오","뱅크샐러드",
          "금융위", "망분리", "금융감독원", "금감원", "정부부처", "정책", 
          "시중은행", "인뱅3사", "이복현", "핀테크",
          "전세대출", "주담대",
          "금융권", "한은", "한국은행"]
        self.non_collect_keywords = ["채용"]
        self.news_df =   self.collect()
        self.client = OpenAI(api_key=self.key)


    
    def run(self):
        # Collect news date every hour 
        for i in range(144) :
            try : 
                new_df = self.collect()
                old_df = self.news_df 
                idxs = []
                for idx, value in enumerate(new_df.url) :
                    if value not in old_df.url.values :
                        idxs.append(idx)

                new_df = pd.concat([old_df, new_df.iloc[idxs]], ignore_index=True)
                self.news_df = new_df
                self.save()
                print("Saved time : ",  self.save_date,  i)
            except : 
                print("Error : ", datetime.now())
            time.sleep(3600) 

    def collect(self):
        urls = get_naverlinks(url="https://news.naver.com/breakingnews/section/101/259?date={}".format(self.save_date))
        news_list = [fetch_article_details(url) for url in urls]
        news_df = pd.DataFrame(news_list)
        TF = news_df.content.map(lambda x : True if any(keyword in x for keyword in self.collect_keywords) else False)
        news_df = news_df[TF]
        TF = news_df.content.map(lambda x : False if any(keyword in x for keyword in self.non_collect_keywords) else True)
        news_df = news_df[TF]
        news_df["shorturl"] = news_df["url"].map(lambda x : get_shorturl(x,self.credential))
        return news_df 
    
    def save(self):
        fname = self.fpath +  self.save_date + ".csv"
        if self.isexist_flag :
            self.news_df.to_csv(fname, index=False)
        else :
            # Check previous 
            if os.path.isfile(fname):
                old_df = pd.read_csv(fname)
                new_df = self.news_df 
                idxs = []

                for idx, value in enumerate(new_df.url) :
                    if value not in old_df.url.values :
                        idxs.append(idx)

                new_df = pd.concat([old_df, new_df.iloc[idxs]], ignore_index=True)
                self.news_df = new_df
                self.isexist_flag = True 
                self.news_df.to_csv(fname, index=False)
            else : # 24 시간이 지나서 새로운 파일 생성 해야될 때
                self.news_df.to_csv(fname, index=False)

            
            

if __name__ == "__main__":
    bot = main_bot()
    #bot = parse_bot(cred_path)
    bot.run()
