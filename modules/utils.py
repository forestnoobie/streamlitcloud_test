import os
import sys
import yaml
import random
from pathlib import Path

import urllib.request
import requests
from datetime import datetime, timedelta
import itertools
 
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import jinja2

from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

import time
from bs4 import BeautifulSoup


# # Short URL
# credential = yaml.safe_load(Path('../credential.yaml').read_text())


def get_shorturl(original_url, credential):
    # client_id = credential['naver_short_url']['client_id']
    # client_secret = credential['naver_short_url']['client_secret']
    # encText = urllib.parse.quote(original_url)
    # data = "url=" + encText
    # url = "https://openapi.naver.com/v1/util/shorturl"
    # request = urllib.request.Request(url)
    # request.add_header("X-Naver-Client-Id",client_id)
    # request.add_header("X-Naver-Client-Secret",client_secret)
    # response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    # rescode = response.getcode()
    # if(rescode==200):
    #     response_body = response.read()
    # else:
    #     print("Error Code:" + rescode)

    # response_dict = eval(response_body.decode('utf-8'))
    #return response_dict['result']['url']

    # temp
    return original_url
    
# Crawling
def get_crawl_result(url):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()), options=options)
    
    driver.get(url)
    while True:
        try:
            # "기사 더보기" 버튼 찾기
            more_button = driver.find_element(By.XPATH, '//*[@id="newsct"]/div[2]/div/div[2]/a')  # 버튼의 클래스가 btn_more
            more_button.click()  # 더보기 버튼 클
            #print("Clicked")
            time.sleep(0.1)  # 페이지 로드 대기
        except:
            #print("더 이상 '기사 더보기' 버튼이 없습니다.")
            break
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup
    
# Get URL
def get_naverlinks(url, title_must="", crawling=True):
    urls = []

    if crawling :
        print("Crawling")
        soup = get_crawl_result(url)
        pass
    else :
        response = requests.get(url) 
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            
    articles = soup.find_all('a', class_='sa_thumb_link') 
    for article in articles:
        article_link = article.get('href')
        if article_link.startswith('https://n.news.naver.com'):
            urls.append(article_link)
    print("Total numbers ", len(urls))  
    return urls


# Get Information
def fetch_article_details(article_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(article_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # title
        title_tag = soup.find('h2', {'id': 'title_area'})
        title = title_tag.text.strip() if title_tag else "No Title"
        
        # time
        time_tag = soup.find('span', {'class': 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'})
        time = time_tag.text.strip() if time_tag else "No Time"
        if time_tag : 
            time = time_to_24(time)
        # publisher
        publisher_tag = soup.find('meta', {'property': 'og:article:author'})
        publisher = publisher_tag['content'].split('|')[0].strip() if publisher_tag else "No Publisher"
        
        # content
        content_tag = soup.find('div', {'id': 'newsct_article'})
        content = content_tag.text.strip() if content_tag else "No Content"

        return {
            'title': title,
            'time': time,
            'publisher': publisher,
            'content': content,
            'url': article_url
        }
    else:
        print(f"Failed to fetch article details for URL: {article_url}")
        return None

def time_to_24(time_data) : 
    date, noon, time = time_data.split(" ")
    hour , minute = time.split(":")
    if noon == "오후" and hour != 12:
        hour = str(int(hour) +  12)
    
    new_time = date + " " + hour + ":" + minute
    return new_time

def check_date(time, checktime) :
    time_data = time.split(" ")[0]
    if time_data == checktime :
        return True 
    else :
        return False
        


def weeknum_to_weekday(weeknum):
    weekdays = {6: "일", 0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토"}
    return weekdays.get(weeknum)

# Crawling study website

def get_study(publisher="byline"):
    # URL of the website
    articles = []
    if publisher == "byline":
        url = "https://byline.network/post_curation/main-top/page/1/"
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        soup_all =  soup.find_all('div', class_='entry-content-wrap')
        for article in soup_all:
            title_tag = article.find_all('a')[1].text.strip()
            link_tag = article.find_all('a')[1].get('href').strip()
            time_tag = article.time.get('datetime')
            articles.append({'title': title_tag, 'time': time_tag, 
            'url': link_tag, "publisher" : "바이라인네트워크"})
    return articles


def get_study_format(articles, credential):
    random.seed(1)
    dd_output = ''
    # To Do random -> rule
    article = random.choice(articles)
    dd_output += """○ {}
출처 : {}
- {}
""".format(article['title'],
article['publisher'] , get_shorturl(article['url'], credential))   
    print(dd_output)
    return dd_output  

# jinga

def get_output_format(df, status="", max_num=3):
    dd_output = ''
    if status: 
        df = df[df['class']==status]
    for record in df.to_dict('records')[:max_num]:
        dd_output += """○ {} {} 
출처 : {}
 - {}
▶ {}
    """.format(record['title'],"★" * int(record['importance']), 
    record['publisher'] , record['shorturl'], 
    record['summary']) + "\n\n" 

    return dd_output   


def get_jinga(status, trend, study, target_date=''):

    if not target_date :
        target_date = datetime.now() + timedelta(days=1)
    

    environment = jinja2.Environment()
    template = environment.from_string(
        """
{{month}}월 {{day}}일 ({{weekday}})  『디지털 데일리』 입니다.

(1) 금융권 Digital 현황
(2) 금융권 Digital 동향 및 정책
(3) Digital 스터디 365

********************************** 
(1) 금융권 Digital 현황
**********************************

{{status}}

********************************** 
(2) 금융권 Digital 동향 및 정책
********************************** 

{{trend}}

**********************************
(3) Digital 스터디 365
**********************************

{{study}}
        """
    )
    response = template.render(month=target_date.month, day=target_date.day,
                    weekday=weeknum_to_weekday(target_date.weekday()), 
                    status=status,
                    trend=trend,
                    study=study)

    return response

