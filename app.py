import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat

# --- 한글 폰트 설정 ---
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = 'sans-serif' 

st.set_page_config(page_title="설비 이상 진단 대시보드", layout="wide")

@st.cache_data
def load_mat_data(file):
    return loadmat(file)

def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    crest_factor = peak / rms if rms > 0 else np.nan
    return {"rms": rms, "peak": peak, "kurtosis": kurtosis, "crest_factor": crest_factor}

st.title("⚙️ 설비 이상 분석 및 CBM 진단 대시보드")

with st.sidebar:
    st.header("1. 데이터셋 설정")
    uploaded_normal = st.file_uploader("정상 데이터 (.mat)", type="mat")
    uploaded_fault = st.file_uploader("이상 데이터 (.mat)", type="mat")
    fs = st.number_input("샘플링 주파수 (Hz)", value=12000)

if uploaded_normal and uploaded_fault:
    mat_n, mat_f = load_mat_data(uploaded_normal), load_mat_data(uploaded_fault)
    n_key = [k for k in mat_n.keys() if not k.startswith("__")][0]
    f_key = [k for k in mat_f.keys() if not k.startswith("__")][0]
    
    sig_n, sig_f = mat_n[n_key].ravel(), mat_f[f_key].ravel()
    
    # 두 신호의 길이를 최소값으로 맞춰서 FFT 축 불일치 방지
    min_len = min(len(sig_n), len(sig_f))
    sig_n, sig_f = sig_n[:min_len], sig_f[:min_len]

    # 1. 시간 영역 및 주파수 영역 비교
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("시간 영역 파형")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(sig_n[:1000], label='Normal', alpha=0.7)
        ax.plot(sig_f[:1000], label='Fault', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("주파수 영역 (FFT)")
        yf_n = np.abs(rfft(sig_n))
        yf_f = np.abs(rfft(sig_f))
        # 데이터 길이를 명확하게 지정하여 주파수 축 생성
        xf = rfftfreq(min_len, 1/fs)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(xf, yf_n, label='Normal', alpha=0.5)
        ax.plot(xf, yf_f, label='Fault', alpha=0.5)
        ax.set_xlim(0, fs/2)
        ax.legend()
        st.pyplot(fig)

    # 2. 특징값 비교
    st.header("3. 특징값 분석 (RMS, Kurtosis, CF)")
    feat_n, feat_f = calculate_features(sig_n), calculate_features(sig_f)
    df = pd.DataFrame([feat_n, feat_f], index=["정상", "이상"])
    st.table(df)

    # 3. CBM 정비 의사결정 섹션
    st.header("4. CBM 기반 정비 의사결정")
    rms_threshold = feat_n["rms"] * 3
    
    st.write("### 진단 리포트")
    if feat_f["rms"] > rms_threshold:
        st.error(f"⚠️ 상태 주의: RMS 값이 정상 대비 3배 이상 증가했습니다 (임계치: {rms_threshold:.2f}).")
        st.write("- **권장 조치:** 즉시 설비 정밀 점검 필요 (베어링 마모 가능성).")
    else:
        st.success("✅ 상태 양호: 정상 범위 내 유지 중.")
        st.write("- **권장 조치:** 정기 예방 정비 일정 유지.")
        
    st.write("---")
    st.caption("참고: 이 결과는 시간/주파수 특징 추출을 바탕으로 한 시범 진단 결과입니다.")
else:
    st.info("사이드바에서 정상과 이상 데이터(.mat)를 모두 업로드해주세요.")
