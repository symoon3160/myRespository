# MAT Vibration Analysis Dashboard

정상 `.mat` 데이터와 비정상 `.mat` 데이터를 업로드해 진동 신호를 비교 분석하는 Streamlit 대시보드입니다.

## 주요 기능

- 정상/비정상 MAT 파일 업로드
- MAT 내부 숫자형 신호 변수 자동 탐색 및 선택
- 시간 영역 파형 비교
- FFT 주파수 영역 비교
- RMS, Peak, Kurtosis, Crest Factor 등 특징값 계산
- 윈도우 기반 추세 분석
- 규칙 기반 상태 진단: 정상, 주의, 위험
- Markdown 리포트 다운로드

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더를 GitHub 저장소에 업로드합니다.
2. Streamlit Community Cloud에서 새 앱을 생성합니다.
3. 저장소와 브랜치를 선택하고 main file path를 `app.py`로 지정합니다.
4. Deploy를 누릅니다.

## 입력 데이터

- MATLAB `.mat` 파일을 사용합니다.
- CWRU 데이터처럼 `DE_time`, `FE_time`, `BA_time` 등이 들어 있는 파일은 자동으로 우선 선택됩니다.
- MATLAB v7.3 HDF5 형식도 `h5py`를 통해 가능한 범위에서 읽습니다.

## 진단 기준

기본 규칙은 다음과 같습니다.

- RMS: 정상 구간 평균 + 3 sigma 초과
- Kurtosis: 5 이상
- Crest Factor: 4 이상

이 기준은 초기 선별용입니다. 실제 설비 적용 시에는 설비별 정상 운전 데이터를 충분히 모아 기준값을 재설정하세요.
