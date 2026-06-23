from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat


st.set_page_config(
    page_title="MAT Bearing Vibration Dashboard",
    page_icon="📈",
    layout="wide",
)


SIGNAL_KEY_HINTS = (
    "DE_time",
    "FE_time",
    "BA_time",
    "B_time",
    "IR_time",
    "OR_time",
    "Normal",
    "Fault",
    "time",
    "vibration",
    "signal",
)


@dataclass(frozen=True)
class SignalBundle:
    label: str
    file_name: str
    key: str
    signal: np.ndarray


def _as_numeric_1d(value: Any) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.dtype.kind not in "biufc":
        return None

    arr = np.real(arr).astype(float, copy=False).squeeze()
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        longest_axis = int(np.argmax(arr.shape))
        arr = np.moveaxis(arr, longest_axis, 0).reshape(arr.shape[longest_axis], -1)[:, 0]

    arr = arr[np.isfinite(arr)]
    if arr.size < 16:
        return None
    return arr.ravel()


def _load_legacy_mat(file_bytes: bytes) -> dict[str, np.ndarray]:
    raw = loadmat(BytesIO(file_bytes), squeeze_me=True, struct_as_record=False)
    signals: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = _as_numeric_1d(value)
        if arr is not None:
            signals[key] = arr
    return signals


def _load_hdf5_mat(file_bytes: bytes) -> dict[str, np.ndarray]:
    signals: dict[str, np.ndarray] = {}

    with h5py.File(BytesIO(file_bytes), "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.dtype.kind not in "biufc":
                return
            arr = _as_numeric_1d(obj[()])
            if arr is not None:
                signals[name] = arr

        handle.visititems(visitor)

    return signals


@st.cache_data(show_spinner=False)
def load_mat_signals(file_bytes: bytes) -> dict[str, np.ndarray]:
    try:
        signals = _load_legacy_mat(file_bytes)
    except NotImplementedError:
        signals = _load_hdf5_mat(file_bytes)
    except ValueError as exc:
        if "Unknown mat file type" not in str(exc):
            raise
        signals = _load_hdf5_mat(file_bytes)

    if not signals:
        raise ValueError("분석 가능한 숫자형 1차원 신호를 찾지 못했습니다.")
    return signals


def default_signal_key(signals: dict[str, np.ndarray]) -> str:
    ranked = sorted(
        signals,
        key=lambda key: (
            not any(hint.lower() in key.lower() for hint in SIGNAL_KEY_HINTS),
            -signals[key].size,
            key.lower(),
        ),
    )
    return ranked[0]


def trim_signal(signal: np.ndarray, max_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size <= max_samples:
        return signal
    return signal[:max_samples]


def calculate_features(signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=float).ravel()
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    mean_abs = float(np.mean(np.abs(signal)))
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(signal)),
        "kurtosis": float(stats.kurtosis(signal, fisher=False, nan_policy="omit")),
        "skewness": float(stats.skew(signal, nan_policy="omit")),
        "crest_factor": float(peak / rms) if rms > 0 else np.nan,
        "shape_factor": float(rms / mean_abs) if mean_abs > 0 else np.nan,
        "impulse_factor": float(peak / mean_abs) if mean_abs > 0 else np.nan,
    }


def compute_fft(signal: np.ndarray, fs: float) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=float).ravel()
    signal = signal - np.mean(signal)
    window = np.hanning(signal.size)
    spectrum = np.abs(rfft(signal * window)) / signal.size
    freq = rfftfreq(signal.size, 1 / fs)
    return pd.DataFrame({"frequency_hz": freq, "amplitude": spectrum})


def window_features(signal: np.ndarray, fs: float, window_sec: float, step_sec: float) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=float).ravel()
    window = max(16, int(fs * window_sec))
    step = max(1, int(fs * step_sec))
    rows = []
    for start in range(0, signal.size - window + 1, step):
        segment = signal[start : start + window]
        rows.append({"time_sec": start / fs, **calculate_features(segment)})
    return pd.DataFrame(rows)


def diagnose_fault_windows(
    normal_windows: pd.DataFrame,
    fault_windows: pd.DataFrame,
    rms_sigma: float,
    kurtosis_threshold: float,
    crest_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rms_threshold = float(normal_windows["rms"].mean() + rms_sigma * normal_windows["rms"].std(ddof=0))

    rows = []
    for _, row in fault_windows.iterrows():
        reasons = []
        if row["rms"] > rms_threshold:
            reasons.append("RMS 증가")
        if row["kurtosis"] > kurtosis_threshold:
            reasons.append("Kurtosis 증가")
        if row["crest_factor"] > crest_threshold:
            reasons.append("Crest Factor 증가")

        if len(reasons) >= 2:
            level = "위험"
        elif reasons:
            level = "주의"
        else:
            level = "정상"

        rows.append({**row.to_dict(), "diagnosis": level, "reason": ", ".join(reasons) or "-"})

    thresholds = {
        "rms_threshold": rms_threshold,
        "kurtosis_threshold": float(kurtosis_threshold),
        "crest_threshold": float(crest_threshold),
    }
    return pd.DataFrame(rows), thresholds


def signal_selector(label: str, uploaded_file: Any, max_samples: int) -> SignalBundle | None:
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    signals = load_mat_signals(file_bytes)
    options = list(signals)
    default_index = options.index(default_signal_key(signals))

    key = st.selectbox(
        f"{label} 신호 변수",
        options=options,
        index=default_index,
        format_func=lambda name: f"{name} ({signals[name].size:,} samples)",
    )
    return SignalBundle(
        label=label,
        file_name=uploaded_file.name,
        key=key,
        signal=trim_signal(signals[key], max_samples),
    )


def plot_waveform(normal: SignalBundle, fault: SignalBundle, fs: float, seconds: float) -> go.Figure:
    fig = go.Figure()
    for bundle, color in ((normal, "#2563eb"), (fault, "#dc2626")):
        n = min(bundle.signal.size, int(fs * seconds))
        x = np.arange(n) / fs
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=bundle.signal[:n],
                mode="lines",
                name=bundle.label,
                line={"color": color, "width": 1.2},
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend_title_text="",
    )
    return fig


def plot_fft_compare(normal: SignalBundle, fault: SignalBundle, fs: float, max_freq: float) -> go.Figure:
    fig = go.Figure()
    for bundle, color in ((normal, "#2563eb"), (fault, "#dc2626")):
        fft_df = compute_fft(bundle.signal, fs)
        fft_df = fft_df[fft_df["frequency_hz"] <= max_freq]
        fig.add_trace(
            go.Scattergl(
                x=fft_df["frequency_hz"],
                y=fft_df["amplitude"],
                mode="lines",
                name=bundle.label,
                line={"color": color, "width": 1.2},
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        legend_title_text="",
    )
    return fig


def make_report(
    normal: SignalBundle,
    fault: SignalBundle,
    fs: float,
    features: pd.DataFrame,
    thresholds: dict[str, float],
    diagnosis: pd.DataFrame,
) -> str:
    counts = diagnosis["diagnosis"].value_counts()
    total = max(1, len(diagnosis))
    danger_rate = counts.get("위험", 0) / total * 100
    warning_rate = counts.get("주의", 0) / total * 100

    return f"""# MAT 진동 분석 리포트

## 분석 파일
- 정상 데이터: {normal.file_name} / 변수: {normal.key}
- 비정상 데이터: {fault.file_name} / 변수: {fault.key}
- 샘플링 주파수: {fs:,.0f} Hz

## 주요 특징값
{features.to_markdown(index=False)}

## 진단 기준
- RMS 기준: 정상 구간 RMS 평균 + sigma = {thresholds["rms_threshold"]:.6g}
- Kurtosis 기준: {thresholds["kurtosis_threshold"]:.6g}
- Crest Factor 기준: {thresholds["crest_threshold"]:.6g}

## 구간 진단 요약
- 정상: {counts.get("정상", 0)}개
- 주의: {counts.get("주의", 0)}개 ({warning_rate:.1f}%)
- 위험: {counts.get("위험", 0)}개 ({danger_rate:.1f}%)

## 해석
위험 또는 주의 구간이 반복적으로 나타나면 결함성 충격, 회전체 불균형, 정렬 불량,
베어링 손상 가능성을 현장 조건과 함께 확인해야 합니다. 공개 데이터 기반 규칙 진단은
초기 선별용이며, 실제 설비 적용 시에는 설비별 정상 데이터로 기준값을 재보정하는 것이 좋습니다.
"""


st.title("MAT 진동 데이터 이상 분석 대시보드")
st.caption("정상 MAT 파일과 비정상 MAT 파일을 업로드하면 파형, FFT, 특징값, 구간별 진단을 비교합니다.")

with st.sidebar:
    st.header("분석 설정")
    fs = st.number_input("샘플링 주파수 (Hz)", min_value=1000.0, value=12000.0, step=100.0)
    max_samples = st.number_input(
        "최대 분석 샘플 수",
        min_value=1_000,
        max_value=2_000_000,
        value=120_000,
        step=10_000,
    )
    waveform_seconds = st.slider("파형 표시 구간 (초)", min_value=0.02, max_value=2.0, value=0.2, step=0.02)
    max_fft_freq = st.slider("FFT 최대 주파수 (Hz)", min_value=100.0, max_value=float(fs / 2), value=min(3000.0, float(fs / 2)), step=100.0)
    window_sec = st.slider("진단 윈도우 (초)", min_value=0.02, max_value=2.0, value=0.2, step=0.02)
    step_sec = st.slider("진단 이동 간격 (초)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    st.divider()
    rms_sigma = st.slider("RMS 기준 sigma", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
    kurtosis_threshold = st.number_input("Kurtosis 기준", min_value=1.0, value=5.0, step=0.5)
    crest_threshold = st.number_input("Crest Factor 기준", min_value=1.0, value=4.0, step=0.5)

left, right = st.columns(2)
with left:
    normal_file = st.file_uploader("정상 MAT 데이터", type=["mat"], key="normal_mat")
with right:
    fault_file = st.file_uploader("비정상 MAT 데이터", type=["mat"], key="fault_mat")

if normal_file is None or fault_file is None:
    st.info("정상 MAT 파일과 비정상 MAT 파일을 각각 업로드하면 분석이 시작됩니다.")
    st.stop()

try:
    selector_left, selector_right = st.columns(2)
    with selector_left:
        normal_bundle = signal_selector("정상", normal_file, int(max_samples))
    with selector_right:
        fault_bundle = signal_selector("비정상", fault_file, int(max_samples))
except Exception as exc:
    st.error(f"MAT 파일을 읽는 중 문제가 발생했습니다: {exc}")
    st.stop()

if normal_bundle is None or fault_bundle is None:
    st.stop()

feature_df = pd.DataFrame(
    [
        {"state": "정상", **calculate_features(normal_bundle.signal)},
        {"state": "비정상", **calculate_features(fault_bundle.signal)},
    ]
)
ratio_df = feature_df.set_index("state").T
ratio_df["fault_to_normal_ratio"] = ratio_df["비정상"] / ratio_df["정상"].replace(0, np.nan)

normal_windows = window_features(normal_bundle.signal, fs, window_sec, step_sec)
fault_windows = window_features(fault_bundle.signal, fs, window_sec, step_sec)

if normal_windows.empty or fault_windows.empty:
    st.warning("신호 길이가 진단 윈도우보다 짧습니다. 윈도우 시간을 줄이거나 더 긴 신호를 업로드하세요.")
    st.stop()

diagnosis_df, threshold_values = diagnose_fault_windows(
    normal_windows,
    fault_windows,
    rms_sigma,
    kurtosis_threshold,
    crest_threshold,
)

counts = diagnosis_df["diagnosis"].value_counts()
total_windows = len(diagnosis_df)
danger_rate = counts.get("위험", 0) / total_windows * 100
warning_rate = counts.get("주의", 0) / total_windows * 100

metric_cols = st.columns(4)
metric_cols[0].metric("정상 샘플", f"{normal_bundle.signal.size:,}")
metric_cols[1].metric("비정상 샘플", f"{fault_bundle.signal.size:,}")
metric_cols[2].metric("주의 구간", f"{counts.get('주의', 0):,}", f"{warning_rate:.1f}%")
metric_cols[3].metric("위험 구간", f"{counts.get('위험', 0):,}", f"{danger_rate:.1f}%")

tab_wave, tab_features, tab_trend, tab_report = st.tabs(["파형/FFT", "특징값", "구간 진단", "리포트"])

with tab_wave:
    st.subheader("시간 영역 파형")
    st.plotly_chart(plot_waveform(normal_bundle, fault_bundle, fs, waveform_seconds), use_container_width=True)

    st.subheader("주파수 영역 FFT")
    st.plotly_chart(plot_fft_compare(normal_bundle, fault_bundle, fs, max_fft_freq), use_container_width=True)

with tab_features:
    st.subheader("전체 신호 특징값")
    st.dataframe(feature_df, use_container_width=True)

    compare_cols = ["rms", "peak", "kurtosis", "crest_factor", "shape_factor", "impulse_factor"]
    long_features = feature_df.melt(id_vars="state", value_vars=compare_cols, var_name="feature", value_name="value")
    fig = px.bar(long_features, x="feature", y="value", color="state", barmode="group", height=390)
    fig.update_layout(margin={"l": 20, "r": 20, "t": 35, "b": 20}, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("비정상 / 정상 비율")
    st.dataframe(ratio_df, use_container_width=True)

with tab_trend:
    trend_df = pd.concat(
        [
            normal_windows.assign(state="정상"),
            fault_windows.assign(state="비정상"),
        ],
        ignore_index=True,
    )

    st.subheader("구간별 특징값 추세")
    trend_feature = st.selectbox("표시할 특징값", ["rms", "kurtosis", "crest_factor", "peak", "std"])
    fig = px.line(trend_df, x="time_sec", y=trend_feature, color="state", height=360)
    fig.update_layout(margin={"l": 20, "r": 20, "t": 35, "b": 20}, xaxis_title="Time (s)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("비정상 데이터 구간 진단")
    st.dataframe(
        diagnosis_df[["time_sec", "rms", "kurtosis", "crest_factor", "diagnosis", "reason"]],
        use_container_width=True,
    )

    fig = px.histogram(diagnosis_df, x="diagnosis", color="diagnosis", height=320)
    fig.update_layout(margin={"l": 20, "r": 20, "t": 35, "b": 20}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_report:
    report = make_report(normal_bundle, fault_bundle, fs, feature_df, threshold_values, diagnosis_df)
    st.markdown(report)
    st.download_button(
        "리포트 다운로드",
        data=report.encode("utf-8"),
        file_name="mat_vibration_report.md",
        mime="text/markdown",
    )
