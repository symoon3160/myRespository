import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib as mpl

# --- 한글 폰트 설정 (koreanize-matplotlib 대신 설정) ---
# 나눔 폰트가 설치되어 있지 않은 환경을 대비해 기본 폰트 설정
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = 'sans-serif' 

# 페이지 설정
st.set_page_config(page_title="설비 이상 진단 대시보드", layout="wide")

# --- 캐시된 데이터 로드 함수 ---
@st.cache_data
def load_mat_data(file):
    return loadmat(file)

# --- 특징값 계산 함수 ---
def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    crest_factor = peak / rms if rms > 0 else np.nan
    return {
        "rms": rms, "peak": peak, "kurtosis": kurtosis, "crest_factor": crest_factor
    }

# --- 메인 화면 구성 ---
st.title("⚙️ 설비 이상 분석 대시보드")

with st.sidebar:
    st.header("1. 데이터셋 설정")
    uploaded_normal = st.file_uploader("정상 데이터 (.mat)", type="mat")
    uploaded_fault = st.file_uploader("이상 데이터 (.mat)", type="mat")
    fs = st.number_input("샘플링 주파수 (Hz)", value=12000)

if uploaded_normal and uploaded_fault:
    mat_n = load_mat_data(uploaded_normal)
    mat_f = load_mat_data(uploaded_fault)
    
    n_key = [k for k in mat_n.keys() if not k.startswith("__")][0]
    f_key = [k for k in mat_f.keys() if not k.startswith("__")][0]
    
    normal_signal = mat_n[n_key].ravel()
    fault_signal = mat_f[f_key].ravel()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("정상 파형")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(normal_signal[:1000])
        st.pyplot(fig)
    with col2:
        st.subheader("이상 파형")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(fault_signal[:1000])
        st.pyplot(fig)

    st.header("3. 특징값 비교")
    feat_n = calculate_features(normal_signal)
    feat_f = calculate_features(fault_signal)
    df = pd.DataFrame([feat_n, feat_f], index=["정상", "이상"])
    st.table(df)

    st.header("4. 상태 진단 결과")
    threshold = 3 * feat_n["rms"]
    if feat_f["rms"] > threshold:
        st.error(f"⚠️ 위험: RMS 값이 기준치({threshold:.2f})를 초과했습니다.")
    else:
        st.success("✅ 현재 정상 범위 내에 있습니다.")
else:
    st.info("사이드바에서 .mat 파일을 업로드해주세요.")
