import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Welding Data Visualization")

# Function to extract CSV files from a ZIP archive
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
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

# Helper function to extract short file names
def shorten_file_name(file_path):
    base_name = os.path.basename(file_path)
    parts = base_name.split("_")
    hhmmss = parts[1]
    nn = parts[-1].split(".")[0]
    return f"{hhmmss}_{nn}"

# Sidebar for file upload and configuration
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

# Main visualization logic
if "metadata" in st.session_state and isinstance(st.session_state["metadata"], dict):
    shortened_file_names = {shorten_file_name(file): file for file in st.session_state["metadata"].keys()}
    sorted_shortened_names = sorted(shortened_file_names.keys())
    selected_files_short = st.sidebar.multiselect("Select CSV files", sorted_shortened_names, key="selected_files")
    selected_files = [shortened_file_names[short_name] for short_name in selected_files_short]
    selected_bead = st.sidebar.number_input("Select Bead Number", min_value=1, value=1, step=1)
    
    # Visualization options
    visualization_option = st.sidebar.selectbox(
        "Select Visualization",
        [
            "Show Spectrogram",
            "Show Frequency Intensity Plot (Line Plot)",
            "Heatmap of Frequency Intensity",
            "Scatter Plot of Frequency Intensity Ratios",
            "Binary Thresholding",
            "Summary Statistics Visualization"
        ]
    )
    
    if selected_files:
        for file in selected_files:
            if selected_bead <= len(st.session_state["metadata"][file]):
                df = pd.read_csv(file)
                start, end = st.session_state["metadata"][file][selected_bead - 1]
                sample_data = df.iloc[start:end, :]
                signal_data = sample_data.iloc[:, 0].values  # Assuming first column is the signal
                
                fs = 10000
                f, t, Sxx = signal.spectrogram(signal_data, fs, nperseg=1024, noverlap=512, nfft=2048)
                Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

                if visualization_option == "Show Spectrogram":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='jet')
                    ax.set_ylim([0, 500])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title(f"Spectrogram - Bead {selected_bead}")
                    st.pyplot(fig)
                
                elif visualization_option == "Show Frequency Intensity Plot (Line Plot)":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(f, np.mean(Sxx, axis=1))
                    ax.set_xlim([0, 500])
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Intensity")
                    ax.set_title(f"Frequency Intensity Plot - Bead {selected_bead}")
                    st.pyplot(fig)
                
                elif visualization_option == "Heatmap of Frequency Intensity":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    heatmap_data = Sxx_dB[(f >= 0) & (f <= 500), :]
                    ax.imshow(heatmap_data, aspect='auto', cmap='jet', extent=[t.min(), t.max(), 0, 500])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title(f"Heatmap of Frequency Intensity - Bead {selected_bead}")
                    st.pyplot(fig)
                
                elif visualization_option == "Scatter Plot of Frequency Intensity Ratios":
                    ratio_400_200 = Sxx[f == 400, :].mean() / Sxx[f == 200, :].mean()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(ratio_400_200)), ratio_400_200)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Intensity Ratio (400Hz/200Hz)")
                    ax.set_title(f"Scatter Plot of Frequency Ratios - Bead {selected_bead}")
                    st.pyplot(fig)
                
                elif visualization_option == "Binary Thresholding":
                    intensity_400 = Sxx[f == 400, :].mean(axis=1)
                    binary = intensity_400 > threshold
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(binary, label="Binary Threshold")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Above Threshold (1=True)")
                    ax.set_title(f"Binary Thresholding - Bead {selected_bead}")
                    st.pyplot(fig)
                
                elif visualization_option == "Summary Statistics Visualization":
                    mean_400 = Sxx[f == 400, :].mean()
                    mean_200 = Sxx[f == 200, :].mean()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(["200Hz", "400Hz"], [mean_200, mean_400])
                    ax.set_ylabel("Mean Intensity")
                    ax.set_title(f"Summary Statistics - Bead {selected_bead}")
                    st.pyplot(fig)
