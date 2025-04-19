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
        
        # Load the first CSV file to get column names
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        # Segment beads based on threshold
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

# Main logic for plotting
if "metadata" in st.session_state and isinstance(st.session_state["metadata"], dict):
    selected_files = st.sidebar.multiselect("Select CSV files", list(st.session_state["metadata"].keys()))
    if selected_files:
        selected_bead = st.sidebar.number_input("Select Bead Number", min_value=1, value=1)
        column_options = df_sample.columns[:2].tolist()
        selected_column = st.sidebar.selectbox("Select Data Column", column_options)
        frequency = st.sidebar.number_input("Enter Frequency (Hz)", min_value=1, value=240)
        
        # Create Plotly figure
        fig = go.Figure()
        
        for selected_file in selected_files:
            df = pd.read_csv(selected_file)
            if selected_bead <= len(st.session_state["metadata"][selected_file]):
                start, end = st.session_state["metadata"][selected_file][selected_bead - 1]
                sample_data = df.iloc[start:end, :]
                
                # Calculate spectrogram
                noverlap = int(noverlap_factor * nperseg)
                f, t, Sxx = spectrogram(
                    sample_data[selected_column].to_numpy(),
                    fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft
                )
                Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
                
                # Extract intensity for the selected frequency
                freq_indices = np.where((f >= frequency - 5) & (f <= frequency + 5))[0]
                if len(freq_indices) > 0:
                    intensity_over_time = np.mean(Sxx_dB[freq_indices, :], axis=0)
                    filename = os.path.basename(selected_file)
                    fig.add_trace(go.Scatter(
                        x=t,
                        y=intensity_over_time,
                        mode='lines',
                        name=f"{filename} ({frequency} Hz)"
                    ))

        # Update Plotly layout
        fig.update_layout(
            title=f"Signal Intensity at {frequency} Hz",
            xaxis_title="Time (s)",
            yaxis_title="Signal Intensity (dB)",
            legend_title="CSV Files",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
