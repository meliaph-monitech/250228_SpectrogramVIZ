import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Signal Intensity Comparison")

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal_values = df[column].to_numpy()
    i = 0
    while i < len(signal_values):
        if signal_values[i] > threshold:
            start = i
            while i < len(signal_values) and signal_values[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        if st.button("Segment Beads"):
            with st.spinner("Segmenting beads..."):
                metadata = {}
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    metadata[file] = segments
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata

fs = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1000, max_value=50000, value=10000)
a = st.sidebar.number_input("nperseg parameter (a)", min_value=256, max_value=4096, value=1024)
b = st.sidebar.number_input("Division Factor (b)", min_value=1, max_value=10, value=4)
c = st.sidebar.number_input("Overlap Ratio (c)", min_value=0.0, max_value=1.0, value=0.99)
d = st.sidebar.number_input("nfft parameter (d)", min_value=512, max_value=8192, value=2048)

# Add inputs for y-axis limits
y_axis_min = st.sidebar.number_input("Y-axis Lower Limit (dB)", value=-100.0)
y_axis_max = st.sidebar.number_input("Y-axis Upper Limit (dB)", value=0.0)

if "metadata" in st.session_state and isinstance(st.session_state["metadata"], dict):
    selected_file = st.sidebar.selectbox("Select a CSV file", list(st.session_state["metadata"].keys()))
    
    if selected_file:
        df = pd.read_csv(selected_file)
        bead_options = list(range(1, len(st.session_state["metadata"][selected_file]) + 1))
        selected_bead = st.sidebar.selectbox("Select a Bead Number", bead_options)
        column_options = df.columns[:2].tolist()
        selected_column = st.sidebar.radio("Select Data Column", column_options)
        
        if "selected_frequencies" not in st.session_state:
            st.session_state["selected_frequencies"] = []
        
        frequency = st.sidebar.number_input("Enter Frequency (Hz)", min_value=1, value=240)
        if st.sidebar.button("Add Frequency"):
            if frequency not in st.session_state["selected_frequencies"]:
                st.session_state["selected_frequencies"].append(frequency)
        
        start, end = st.session_state["metadata"][selected_file][selected_bead - 1]
        sample_data = df.iloc[start:end, :2].values
        
        nperseg = min(a, len(sample_data) // b)
        noverlap = int(c * nperseg)
        nfft = min(d, b ** int(np.ceil(np.log2(nperseg * 2))))
        
        f, t, Sxx = signal.spectrogram(sample_data[:, df.columns.get_loc(selected_column)], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
        
        if st.session_state["selected_frequencies"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for freq in st.session_state["selected_frequencies"]:
                freq_indices = np.where((f >= freq - 5) & (f <= freq + 5))[0]
                if len(freq_indices) > 0:
                    intensity_over_time = np.mean(Sxx_dB[freq_indices, :], axis=0)
                    ax.plot(t, intensity_over_time, label=f"{freq} Hz")
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal Intensities (dB)")
            ax.set_ylim(y_axis_min, y_axis_max)  # Apply y-axis limits
            ax.legend()
            st.pyplot(fig)
        
        if st.sidebar.button("Clear Frequencies"):
            st.session_state["selected_frequencies"] = []
