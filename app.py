import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from modules.scenario import main_bot

st.set_page_config(
    page_title="디지털 데일리",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .dd-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 10px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .dd-card a {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    text-decoration: none;
    line-height: 1.5;
  }
  .dd-card a:hover { text-decoration: underline; }
  .dd-meta { font-size: 12px; color: #888; margin: 6px 0 8px; }
  .dd-summary { font-size: 14px; color: #444; line-height: 1.6; margin: 8px 0; }
  .dd-bar-bg {
    flex: 1; height: 5px; background: #e0e0e0;
    border-radius: 3px; max-width: 160px;
  }
  .dd-bar { height: 100%; border-radius: 3px; }
  .dd-sent-row {
    display: flex; align-items: center; gap: 8px; margin-top: 10px;
  }
  .dd-stars { color: #f5a623; font-size: 18px; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────
def sentiment_meta(score: int):
    if score >= 70:
        return "#2ca02c", "긍정"
    elif score >= 50:
        return "#ff7f0e", "중립"
    else:
        return "#d62728", "부정"


def importance_border(imp: int):
    return {3: "#d62728", 2: "#ff7f0e"}.get(imp, "#4a90d9")


def render_card(record: dict):
    importance = int(record.get("importance", 1))
    sentiment  = int(record.get("sentiment",  50))
    title      = record.get("title", "")
    publisher  = record.get("publisher", "")
    url        = record.get("shorturl", record.get("url", "#"))
    summary    = record.get("summary", "")
    time_str   = record.get("time", "")

    s_color, s_label = sentiment_meta(sentiment)
    border = importance_border(importance)
    stars  = "★" * importance + "☆" * (3 - importance)

    st.markdown(f"""
<div class="dd-card" style="border-left: 4px solid {border};">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
    <a href="{url}" target="_blank">{title}</a>
    <span class="dd-stars">{stars}</span>
  </div>
  <div class="dd-meta">📰 {publisher} &nbsp;|&nbsp; 🕐 {time_str}</div>
  <div class="dd-summary">{summary}</div>
  <div class="dd-sent-row">
    <span style="font-size:11px;color:#888;">긍정도</span>
    <div class="dd-bar-bg">
      <div class="dd-bar" style="width:{sentiment}%;background:{s_color};"></div>
    </div>
    <span style="font-size:11px;color:{s_color};font-weight:500;">{s_label} {sentiment}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    selected_date = st.date_input(
        "날짜 선택",
        value=(datetime.now() + timedelta(days=1)).date(),
        help="분석할 뉴스 날짜를 선택하세요",
    )

    st.divider()
    st.markdown("## 🔍 필터")

    min_importance = st.select_slider(
        "최소 중요도",
        options=[1, 2, 3],
        value=1,
        format_func=lambda x: "★" * x,
    )

    sentiment_range = st.slider("긍정도 범위", 0, 100, (0, 100))

    keyword = st.text_input("🔎 키워드 검색", placeholder="검색어를 입력하세요")

    st.divider()
    st.markdown("## 🔗 커스텀 URL")

    custom_urls_text = st.text_area(
        "URL 입력 (줄바꿈으로 구분)",
        placeholder="https://n.news.naver.com/...",
        height=100,
    )
    use_custom_url = st.checkbox("커스텀 URL 사용", value=False)

    st.divider()
    run_btn   = st.button("🚀 실행", use_container_width=True, type="primary")
    clear_btn = st.button("🗑️ 결과 초기화", use_container_width=True)


# ── Session state init ────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

if clear_btn:
    st.session_state["results"] = None
    st.rerun()


# ── Header ────────────────────────────────────────────────────────
st.markdown("# 📰 디지털 데일리")

with st.expander("주의 사항"):
    st.markdown("""
본 프로그램은 매 시간 단위로 수집된 네이버 금융 뉴스를 기반으로 요약을 진행합니다.
따라서 매시간 마다 새로고침 하면 출력값이 달라질 수 있습니다.

* 실행 버튼을 누르면 프로그램이 실행되며 5~10분 정도의 시간이 소요됩니다.
* 해당 서비스는 개인 계정 및 서버를 사용하고 있으니 공유는 자제 부탁드립니다.
    """)


# ── Run ───────────────────────────────────────────────────────────
if run_btn:
    if use_custom_url and custom_urls_text.strip():
        with open("./urls.txt", "w") as f:
            f.write(custom_urls_text.strip())

    date_str = selected_date.strftime("%Y%m%d")

    try:
        bot = main_bot(True, date=date_str, custom_url=use_custom_url)
    except Exception:
        bot = main_bot(False, date=date_str, custom_url=use_custom_url)

    bot.run()

    st.session_state["results"] = {
        "news_df":      bot.news_df.copy(),
        "study":        getattr(bot, "study", ""),
        "response":     getattr(bot, "_response", ""),
        "response_all": getattr(bot, "_response_all", ""),
        "date":         bot.target_date,
    }


# ── Results ───────────────────────────────────────────────────────
if st.session_state["results"]:
    results = st.session_state["results"]
    news_df = results["news_df"].copy()

    # Apply filters
    fdf = news_df.copy()
    if "importance" in fdf.columns:
        fdf = fdf[fdf["importance"] >= min_importance]
    if "sentiment" in fdf.columns:
        fdf = fdf[fdf["sentiment"].between(sentiment_range[0], sentiment_range[1])]
    if keyword:
        hit = (
            fdf["title"].str.contains(keyword, case=False, na=False)
            | fdf["summary"].str.contains(keyword, case=False, na=False)
        )
        fdf = fdf[hit]
    fdf = fdf.reset_index(drop=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📰 전체 기사", len(news_df))
    if "class" in news_df.columns:
        c2.metric("🏦 금융회사",        len(news_df[news_df["class"] == "금융회사"]))
        c3.metric("📋 정부정책 및 동향", len(news_df[news_df["class"] == "정부정책 및 동향"]))
    else:
        c2.metric("🏦 금융회사", "-")
        c3.metric("📋 정부정책 및 동향", "-")
    avg_sent = round(news_df["sentiment"].mean(), 1) if "sentiment" in news_df.columns else "-"
    c4.metric("😊 평균 긍정도", avg_sent)
    c5.metric("⭐ 필터 결과", len(fdf))

    st.divider()

    # Download buttons
    date_label = results["date"].strftime("%Y%m%d")
    d1, d2, _, _ = st.columns(4)
    d1.download_button(
        "📥 뉴스레터 다운로드",
        results["response"],
        file_name=f"디지털데일리_{date_label}.txt",
        use_container_width=True,
    )
    d2.download_button(
        "📥 전체 기사 다운로드",
        results["response_all"],
        file_name=f"전체기사_{date_label}.txt",
        use_container_width=True,
    )

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏦 금융회사",
        "📋 정부정책 및 동향",
        "📚 Digital 스터디",
        "📄 전체 원문",
    ])

    with tab1:
        fin_df = fdf[fdf["class"] == "금융회사"] if "class" in fdf.columns else pd.DataFrame()
        if fin_df.empty:
            st.info("필터 조건에 맞는 기사가 없습니다.")
        else:
            for _, row in fin_df.iterrows():
                render_card(row.to_dict())

    with tab2:
        pol_df = fdf[fdf["class"] == "정부정책 및 동향"] if "class" in fdf.columns else pd.DataFrame()
        if pol_df.empty:
            st.info("필터 조건에 맞는 기사가 없습니다.")
        else:
            for _, row in pol_df.iterrows():
                render_card(row.to_dict())

    with tab3:
        st.markdown("### 오늘의 Digital 스터디")
        st.markdown(results["study"])

    with tab4:
        st.markdown("### 전체 원문")
        st.code(results["response_all"])

else:
    st.info("사이드바에서 날짜를 선택하고 **🚀 실행** 버튼을 눌러주세요.")
