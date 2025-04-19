import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from scipy.signal import spectrogram
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Signal Intensity Comparison with Plotly")

# Function to extract CSV files from a ZIP file
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

# Function to segment beads based on a threshold
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

# Sidebar for file upload and settings
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

# Sampling and spectrogram parameters
fs = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1000, max_value=50000, value=10000)
nperseg = st.sidebar.number_input("nperseg parameter", min_value=256, max_value=4096, value=1024)
noverlap_factor = st.sidebar.slider("Overlap Ratio", min_value=0.0, max_value=1.0, value=0.5)
nfft = st.sidebar.number_input("nfft parameter", min_value=512, max_value=8192, value=2048)

# Main plot generation
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
        
        noverlap = int(noverlap_factor * nperseg)
        
        f, t, Sxx = spectrogram(sample_data[:, df.columns.get_loc(selected_column)], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

        # Generate Plotly figure
        fig = go.Figure()
        for freq in st.session_state["selected_frequencies"]:
            freq_indices = np.where((f >= freq - 5) & (f <= freq + 5))[0]
            if len(freq_indices) > 0:
                intensity_over_time = np.mean(Sxx_dB[freq_indices, :], axis=0)
                fig.add_trace(go.Scatter(x=t, y=intensity_over_time, mode='lines', name=f"{freq} Hz"))

        fig.update_layout(
            title="Signal Intensity Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Signal Intensity (dB)",
            legend_title="Frequency",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.sidebar.button("Clear Frequencies"):
            st.session_state["selected_frequencies"] = []
