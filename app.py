import streamlit as st
import pandas as pd
import numpy as np

# Ensure matplotlib is installed and imported
import subprocess
import sys
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
import os

st.set_page_config(layout="wide")

st.title("⚙️ 공개 진동 데이터 기반 설비 이상 분석")
st.markdown("--- ")

# --- 0. 라이브러리 불러오기 (이미 노트북에서 불러왔으므로 여기서는 최소한으로)
# koreanize_matplotlib (설치되어 있어야 함)
plt.rcParams["axes.unicode_minus"] = False

# --- Helper Functions (From the original notebook) ---
def calculate_features(signal):
    signal = np.asarray(signal).ravel()
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    crest_factor = peak / rms if rms > 0 else np.nan
    std = np.std(signal)
    mean_abs = np.mean(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": std,
        "rms": rms,
        "peak": peak,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "mean_abs": mean_abs,
    }

def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    signal = signal - np.mean(signal) # Remove DC component
    n = len(signal)
    window = np.hanning(n) # Apply Hanning window
    spectrum = np.abs(rfft(signal * window)) / n
    freq = rfftfreq(n, 1 / fs)
    return freq, spectrum

def plot_time_waveform(signal, fs, title, seconds=0.2):
    n = min(len(signal), int(fs * seconds))
    x = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, signal[:n])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    return fig

def plot_fft(signal, fs, title, max_freq=1000):
    freq, spectrum = compute_fft(signal, fs)
    mask = freq <= max_freq
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freq[mask], spectrum[mask])
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    return fig

def window_features(signal, fs, window_sec=0.2, step_sec=0.1):
    signal = np.asarray(signal).ravel()
    window = int(fs * window_sec)
    step = int(fs * step_sec)
    rows = []
    # Ensure there's at least one full window to process
    if len(signal) < window: 
        return pd.DataFrame() # Return empty DataFrame if signal is too short

    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        rows.append({
            "time_sec": start / fs,
            **calculate_features(seg),
        })
    return pd.DataFrame(rows)

def diagnose(row, rms_threshold, kurtosis_threshold, crest_threshold):
    reasons = []
    if row["rms"] > rms_threshold:
        reasons.append("RMS 증가")
    if row["kurtosis"] > kurtosis_threshold:
        reasons.append("충격성 증가")
    if row["crest_factor"] > crest_threshold:
        reasons.append("Crest Factor 증가")

    if len(reasons) >= 2:
        return "위험", ", ".join(reasons)
    if len(reasons) == 1:
        return "주의", reasons[0]
    return "정상", "-"


# --- Streamlit App --- 
st.header("1. 데이터셋 선택 및 업로드")

DATASET_NAME = st.text_input("데이터셋 명", "CWRU Bearing Dataset")
DATASET_URL = st.text_input("데이터 출처 URL", "https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets?resource=download")
FS = st.number_input("샘플링 주파수 (Hz)", min_value=1, value=12000, step=1000)

st.subheader("데이터 파일 업로드 (.mat 파일)")

uploaded_normal_file = st.file_uploader("정상 데이터 파일 (예: NormalData.mat)", type="mat")
uploaded_fault_file = st.file_uploader("이상 데이터 파일 (예: ErrorData.mat)", type="mat")

normal_signal = None
fault_signal = None

if uploaded_normal_file and uploaded_fault_file:
    try:
        # Load normal data
        mat_normal = loadmat(uploaded_normal_file)
        normal_keys = [k for k in mat_normal.keys() if not k.startswith("__")]
        st.write(f"정상 MAT 변수 목록: {normal_keys}")
        normal_signal_key = st.selectbox("정상 신호로 사용할 변수 선택", normal_keys, key='normal_select')
        normal_signal = mat_normal[normal_signal_key].ravel()

        # Load fault data
        mat_fault = loadmat(uploaded_fault_file)
        fault_keys = [k for k in mat_fault.keys() if not k.startswith("__")]
        st.write(f"이상 MAT 변수 목록: {fault_keys}")
        fault_signal_key = st.selectbox("이상 신호로 사용할 변수 선택", fault_keys, key='fault_select')
        fault_signal = mat_fault[fault_signal_key].ravel()

        st.success("데이터 로드 완료!")
        st.write(f"정상 신호 길이: {len(normal_signal)}")
        st.write(f"이상 신호 길이: {len(fault_signal)}")

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.stop()

if normal_signal is not None and fault_signal is not None:
    st.markdown("--- ")
    st.header("2. 시간 영역 파형 비교")
    st.pyplot(plot_time_waveform(normal_signal, FS, "정상 진동 신호"))
    st.pyplot(plot_time_waveform(fault_signal, FS, "이상 진동 신호"))
    st.markdown("**확인 질문:**")
    st.markdown("- 이상 신호에 충격성 피크가 보이는가?")
    st.markdown("- 진폭이 정상보다 커졌는가?")
    st.markdown("- 반복적인 충격 패턴이 있는가?")

    st.markdown("--- ")
    st.header("3. 시간 영역 특징값 계산")
    feature_df = pd.DataFrame([
        {"state": "normal", **calculate_features(normal_signal)},
        {"state": "fault", **calculate_features(fault_signal)},
    ])
    st.dataframe(feature_df)

    plot_cols = ["rms", "peak", "kurtosis", "crest_factor"]
    fig, ax = plt.subplots(figsize=(10, 4))
    feature_df.set_index("state")[plot_cols].T.plot(kind="bar", ax=ax)
    ax.set_title("정상/이상 특징값 비교")
    ax.set_ylabel("Feature value")
    ax.set_xticks(rotation=0)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    st.markdown("--- ")
    st.header("4. 주파수 영역 분석")
    st.pyplot(plot_fft(normal_signal, FS, "정상 신호 FFT"))
    st.pyplot(plot_fft(fault_signal, FS, "이상 신호 FFT"))
    st.markdown("**확인 질문:**")
    st.markdown("- 정상 신호와 이상 신호의 주요 주파수가 다른가?")
    st.markdown("- 이상 신호에서 고주파 성분이 증가하는가?")
    st.markdown("- 회전 주파수 또는 결함 주파수로 의심되는 피크가 있는가?")

    st.markdown("--- ")
    st.header("5. 구간별 특징값 추세 분석")
    st.info("신호를 일정 구간으로 나누어 RMS, Kurtosis, Crest Factor의 변화를 확인합니다.")

    # Use slider for window and step settings
    window_sec = st.slider("Window Duration (seconds)", 0.01, 1.0, 0.2, 0.01)
    step_sec = st.slider("Step Duration (seconds)", 0.01, 1.0, 0.1, 0.01)

    normal_win = window_features(normal_signal, FS, window_sec=window_sec, step_sec=step_sec)
    fault_win = window_features(fault_signal, FS, window_sec=window_sec, step_sec=step_sec)

    if not normal_win.empty and not fault_win.empty:
        normal_win["state"] = "normal"
        fault_win["state"] = "fault"
        trend_df = pd.concat([normal_win, fault_win], ignore_index=True)

        st.dataframe(trend_df.head())

        for col in ["rms", "kurtosis", "crest_factor"]:
            fig, ax = plt.subplots(figsize=(12, 3))
            for state, group in trend_df.groupby("state"):
                ax.plot(group["time_sec"], group[col], label=state)
            ax.set_title(f"구간별 {col} 추세")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.markdown("--- ")
        st.header("6. 규칙 기반 상태진단 기준 제안")
        st.info("아래 기준은 예시입니다. 실제 현장에서는 설비 종류, 센서 위치, 부하 조건, 회전수에 따라 기준을 별도로 정해야 합니다.")

        # Dynamically set thresholds
        if not normal_win.empty:
            normal_baseline = normal_win[["rms", "kurtosis", "crest_factor"]].agg(["mean", "std"])

            rms_threshold = normal_baseline.loc["mean", "rms"] + 3 * normal_baseline.loc["std", "rms"]
            st.write(f"**RMS 주의 기준:** 정상 RMS 평균 + 3σ = {rms_threshold:.4f}")
        else:
            rms_threshold = 0 # Default if no normal data
            st.write("정상 데이터가 없어 RMS 주의 기준을 설정할 수 없습니다.")

        kurtosis_threshold = st.number_input("**Kurtosis 주의 기준:**", value=5.0, step=0.5)
        crest_threshold = st.number_input("**Crest Factor 주의 기준:**", value=4.0, step=0.5)

        diagnosis = fault_win.copy()
        diagnosis[["diagnosis", "reason"]] = diagnosis.apply(
            lambda row: pd.Series(diagnose(row, rms_threshold, kurtosis_threshold, crest_threshold)),
            axis=1,
        )

        st.subheader("이상 데이터 진단 결과 (상위 20개)")
        st.dataframe(diagnosis[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]].head(20))
        st.write("**진단 결과 분포:**")
        st.dataframe(diagnosis["diagnosis"].value_counts().reset_index()))

        st.markdown("--- ")
        st.header("7. CBM 관점의 의사결정 작성")
        st.markdown("**질문:**")
        st.markdown("1. 정상과 이상 데이터의 가장 큰 차이는 무엇인가?")
        st.markdown("2. 어떤 특징값이 이상 상태를 가장 잘 구분했는가?")
        st.markdown("3. FFT에서 의미 있는 주파수 변화가 있었는가?")
        st.markdown("4. 어떤 기준으로 `정상`, `주의`, `위험`을 나눌 것인가?")
        st.markdown("5. 현장에서는 언제 점검 또는 정비를 지시할 것인가?")
        st.markdown("6. 이 분석을 실제 설비에 적용할 때 한계는 무엇인가?")

        summary_text = f'''
## 공개 진동 데이터 분석 결과 요약

### 사용 데이터
- 데이터셋: {DATASET_NAME}
- 출처: {DATASET_URL}
- 샘플링 주파수: {FS} Hz

### 특징값 비교
- 정상 RMS: {feature_df.loc[feature_df['state']=='normal', 'rms'].iloc[0]:.4f}
- 이상 RMS: {feature_df.loc[feature_df['state']=='fault', 'rms'].iloc[0]:.4f}
- 정상 Kurtosis: {feature_df.loc[feature_df['state']=='normal', 'kurtosis'].iloc[0]:.4f}
- 이상 Kurtosis: {feature_df.loc[feature_df['state']=='fault', 'kurtosis'].iloc[0]:.4f}
- 정상 Crest Factor: {feature_df.loc[feature_df['state']=='normal', 'crest_factor'].iloc[0]:.4f}
- 이상 Crest Factor: {feature_df.loc[feature_df['state']=='fault', 'crest_factor'].iloc[0]:.4f}

### 진단 기준 예시
- RMS 주의 기준: 정상 RMS 평균 + 3σ = {rms_threshold:.4f}
- Kurtosis 주의 기준: {kurtosis_threshold}
- Crest Factor 주의 기준: {crest_threshold}

### CBM 해석
- Kurtosis와 Crest Factor는 정상 데이터 대비 이상 데이터에서 크게 증가하여 충격성 결함을 나타냅니다.
- RMS 값 또한 이상 데이터에서 크게 증가하여 전반적인 진동 에너지 증가를 보여줍니다.
- FFT 분석 결과, 이상 데이터에서 특정 주파수 성분의 진폭이 증가하는 패턴이 관찰될 수 있습니다.
- **점검/정비 의사결정 기준:**
  - RMS, Kurtosis, Crest Factor 중 2개 이상이 주의 기준을 초과하면 '위험' 상태로 판단하여 즉각적인 점검 및 정비를 지시합니다.
  - 1개가 주의 기준을 초과하면 '주의' 상태로 판단하고, 다음 점검 주기까지 모니터링을 강화합니다.
- **현장 적용 한계:** 본 분석은 특정 데이터셋에 기반하며, 실제 설비의 운전 조건(부하, 속도 등)과 베어링의 형상 정보(BPFI, BPFO 등)를 고려한 추가적인 분석이 필요합니다. 또한, 단일 센서 데이터만을 사용했으므로 다중 센서 데이터와 연계한 분석이 더욱 정확한 진단을 가능하게 합니다.
'''
        st.markdown(summary_text)
    else:
        st.warning("구간별 특징값 추세 분석을 위해 충분한 데이터가 필요합니다. Window 및 Step 설정을 확인하거나 파일 크기를 늘려주세요.")
else:
    st.warning("정상 및 이상 데이터 MAT 파일을 모두 업로드해주세요.")
