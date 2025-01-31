import streamlit as st

from modules.scenario import main_bot

from datetime import datetime, timedelta
import re


st.set_page_config(page_title="디지털 데일리", page_icon=":robot:",
                      initial_sidebar_state='collapsed'
)
st.header("디지털 데일리")



with st.expander(("주의 사항")):
    st.markdown((
        """
    본 프로그램은 매 시간 단위로 수집된 네이버 금융 뉴스를 기반으로 요약을 진행합니다.
    따라서 매시간 마다 새로고침 하면 출력값이 달라질 수 있습니다.

    * 실행 버튼을 누르면 프로그램이 실행되며 5~10분 정도의 시간이 소요됩니다.
    * 해당 서비스는 개인 계정 및 서버를 사용하고 있으니 공유는 자제 부탁드립니다.

    """
    ))

## Initialize
if "run" not in st.session_state :
    st.session_state['run'] = False


st.session_state['run'] = False
if st.toggle("실행") : 
    st.session_state['run'] = True

if st.session_state['run'] :
    try : 
        load_option = True
        date =  (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
        print("Initialized")
        bot = main_bot(load_option, date)
        bot.run()
    except :
        load_option = False
        date =  (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
        bot = main_bot(load_option, date)
        bot.run()
    
    

