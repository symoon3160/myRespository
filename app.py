import streamlit as st
import pandas as pd
import gspread
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="팀 예산 관리 시스템", layout="wide")

# 세션 상태에 데이터 초기화 (별도 파일 없이 메모리 관리)
if 'budget_df' not in st.session_state:
    st.session_state.budget_df = pd.DataFrame(columns=["날짜", "팀원", "항목", "금액"])

# 구글 시트 연결 설정 (코드 내 하드코딩 예시)
def get_data_from_sheets():
    try:
        # 실제 사용 시 서비스 계정 키 정보를 여기에 직접 입력하거나 설정 가능
        # service_account_info = { "type": "...", "project_id": "...", ... }
        # credentials = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info)
        # client = gspread.authorize(credentials)
        # sheet = client.open("데이터시트명").sheet1
        # return pd.DataFrame(sheet.get_all_records())
        return st.session_state.budget_df
    except Exception as e:
        st.error("시트 연결 실패: " + str(e))
        return st.session_state.budget_df

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
                new_data = {"날짜": month, "팀원": member, "항목": category, "금액": amount}
                st.session_state.budget_df = pd.concat([st.session_state.budget_df, pd.DataFrame([new_data])], ignore_index=True)
                st.success(f"{member}님의 내역이 저장되었습니다.")
                
    with col2:
        st.subheader("📂 최근 입력 내역")
        st.table(st.session_state.budget_df)

with tab2:
    st.subheader("대시보드")
    if not st.session_state.budget_df.empty:
        # 항목별 합계 계산
        chart_data = st.session_state.budget_df.groupby("항목")["금액"].sum()
        st.bar_chart(chart_data)
        
        # 팀원별 합계
        st.write("### 팀원별 누적 사용액")
        st.bar_chart(st.session_state.budget_df.groupby("팀원")["금액"].sum())
    else:
        st.write("표시할 데이터가 없습니다.")

# 실행 가이드: 
# 1. pip install streamlit pandas gspread
# 2. streamlit run app.py
