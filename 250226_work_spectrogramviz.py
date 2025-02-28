import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Spectrogram VIZ")

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
                metadata = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata

if "metadata" in st.session_state:
    for metadata in st.session_state["metadata"]:
        file = metadata["file"]
        df = pd.read_csv(file)
        fig_width, fig_height = 5, 5
        fig, axes = plt.subplots(2, len(st.session_state["metadata"]), figsize=(fig_width, fig_height), constrained_layout=True)
        
        bead_num = metadata["bead_number"]
        start, end = metadata["start_index"], metadata["end_index"]
        sample_data = df.iloc[start:end, :2].values
        
        fs = 10000  # Sampling frequency
        nperseg = min(1024, len(sample_data) // 4)
        noverlap = int(0.99 * nperseg)
        nfft = min(2048, 4 ** int(np.ceil(np.log2(nperseg * 2))))
        db_scale = 110
        
        f_nir, t_nir, Sxx_nir = signal.spectrogram(sample_data[:, 0], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        Sxx_dB_nir = 20 * np.log10(np.abs(Sxx_nir) + np.finfo(float).eps)
        min_disp_dB_nir = np.max(Sxx_dB_nir) - db_scale
        Sxx_dB_nir[Sxx_dB_nir < min_disp_dB_nir] = min_disp_dB_nir
        
        ax_nir = axes[0, bead_num - 1]
        img_nir = ax_nir.pcolormesh(t_nir, f_nir, Sxx_dB_nir - min_disp_dB_nir, shading='gouraud', cmap='jet')
        ax_nir.set_ylim([0, 500])
        if bead_num == 1:
            ax_nir.set_ylabel("Frequency (Hz)")
        ax_nir.set_xticks([])
        
        f_vis, t_vis, Sxx_vis = signal.spectrogram(sample_data[:, 1], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        Sxx_dB_vis = 20 * np.log10(np.abs(Sxx_vis) + np.finfo(float).eps)
        min_disp_dB_vis = np.max(Sxx_dB_vis) - 90
        Sxx_dB_vis[Sxx_dB_vis < min_disp_dB_vis] = min_disp_dB_vis
        
        ax_vis = axes[1, bead_num - 1]
        img_vis = ax_vis.pcolormesh(t_vis, f_vis, Sxx_dB_vis - min_disp_dB_vis, shading='gouraud', cmap='jet')
        ax_vis.set_ylim([0, 500])
        if bead_num == 1:
            ax_vis.set_ylabel("Frequency (Hz)")
        ax_vis.set_xlabel("Time (s)")
        
    fig.colorbar(img_nir, ax=axes[0, -1], aspect=20)
    fig.colorbar(img_vis, ax=axes[1, -1], aspect=20)
    st.pyplot(fig)
