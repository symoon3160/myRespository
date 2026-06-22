import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# 1. Google Sheets 연결 함수
def get_data():
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets']
    # Streamlit Cloud의 Secrets에서 키를 가져옴
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open("예산시트이름").sheet1
    return pd.DataFrame(sheet.get_all_records())

# 2. UI 구성
st.title("📊 팀 예산 관리 시스템")

tab1, tab2 = st.tabs(["데이터 입력", "전체 대시보드"])

with tab1:
    with st.form("budget_form"):
        member = st.selectbox("팀원 선택", ["부장님", "팀원1", "팀원2"])
        month = st.date_input("날짜")
        category = st.selectbox("항목", ["수선유지비", "비품", "개량공사"])
        amount = st.number_input("금액", min_value=0)
        
        if st.form_submit_button("저장"):
            # 구글 시트에 append 로직 추가 (gspread 사용)
            st.success("저장되었습니다!")

with tab2:
    df = get_data()
    st.metric("전체 누적 사용액", f"{df['금액'].sum():,}원")
    st.bar_chart(df.groupby('항목')['금액'].sum())
```

### 4단계: 깃허브 업로드 및 배포 (보안 핵심)
1.  **깃허브 업로드:** `app.py`, `requirements.txt`만 올립니다. (절대 `service_account.json`을 올리지 마세요!)
2.  **Streamlit Cloud 연동:** * 저장소 연결 후, 배포 설정 화면의 **'Secrets'** 항목에 JSON 파일 내용을 붙여넣습니다:
    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "..."
    ...
