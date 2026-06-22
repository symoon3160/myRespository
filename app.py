import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Google Apps Script 배포 URL (본인의 URL로 교체 필요)
GAS_URL = "https://script.google.com/macros/s/AKfycbxxS1vuHQZIvLnCIKGWZM4sgMB3bqALPkP3bC_lqEwLqE23NoJoWvFyXARO_GSvylBmHQ/exec"

# 페이지 설정
st.set_page_config(page_title="팀 예산 관리 시스템", layout="wide")

# 구글 시트에서 데이터를 가져오는 함수
def get_data_from_sheets():
    try:
        response = requests.get(GAS_URL)
        if response.status_code == 200:
            data = response.json()
            # 첫 번째 행이 헤더인 경우 데이터프레임으로 변환
            if len(data) > 1:
                return pd.DataFrame(data[1:], columns=data[0])
            return pd.DataFrame(columns=["날짜", "팀원", "항목", "금액"])
        else:
            return pd.DataFrame(columns=["날짜", "팀원", "항목", "금액"])
    except Exception as e:
        st.error(f"시트 연결 실패: {e}")
        return pd.DataFrame(columns=["날짜", "팀원", "항목", "금액"])

st.title("📊 팀 예산 관리 시스템")

tab1, tab2 = st.tabs(["데이터 입력", "대시보드"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("budget_form"):
            member = st.selectbox("팀원 선택", ["부장님", "팀원1", "팀원2", "팀원3", "팀원4"])
            month = st.date_input("해당 월").strftime("%Y-%m")
            category = st.selectbox("예산 항목", ["수선유지비", "비품", "개량공사"])
            amount = st.number_input("금액", min_value=0, step=1000)
            submitted = st.form_submit_button("저장")
            
            if submitted:
                payload = {"month": month, "member": member, "category": category, "amount": amount}
                try:
                    response = requests.post(GAS_URL, json=payload)
                    if response.status_code == 200:
                        st.success(f"{member}님의 내역이 구글 시트에 저장되었습니다.")
                    else:
                        st.error("저장 중 오류가 발생했습니다.")
                except Exception as e:
                    st.error(f"서버 연결 오류: {e}")
                
    with col2:
        st.subheader("📂 최근 입력 내역")
        df = get_data_from_sheets()
        st.table(df)

with tab2:
    st.subheader("대시보드")
    df = get_data_from_sheets()
    if not df.empty:
        # 데이터 타입 변환
        df["금액"] = pd.to_numeric(df["금액"])
        
        # 항목별 합계
        chart_data = df.groupby("항목")["금액"].sum()
        st.bar_chart(chart_data)
        
        # 팀원별 합계
        st.write("### 팀원별 누적 사용액")
        st.bar_chart(df.groupby("팀원")["금액"].sum())
    else:
        st.write("표시할 데이터가 없습니다.")

# 실행 가이드: 
# 1. pip install streamlit pandas requests
# 2. streamlit run app.py
