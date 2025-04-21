import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Spectrogram VIZ")

# Function to extract CSV files from a ZIP archive
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):  # Check if the directory exists
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path):  # Ensure it's a file (not a subdirectory)
                os.remove(file_path)
    else:
        os.makedirs(extract_dir)  # Create the directory if it doesn't exist
    
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
    # base_name = os.path.basename(file_path)
    # parts = base_name.split("_")
    # hhmmss = parts[1]  # The second part is always the time (hhmmss)
    # nn = parts[-1].split(".")[0]  # The last part before .csv is the nn
    # return f"{hhmmss}_{nn}"
def shorten_file_name(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    parts = name_without_ext.split("_")
    # Get everything from the third part (index 2) onward, joined by underscores
    trimmed_part = "_".join(parts[2:])
    return trimmed_part

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
    
    if selected_files:
        bead_options = []
        for file in selected_files:
            bead_count = len(st.session_state["metadata"][file])
            bead_options.extend(range(1, bead_count + 1))
        bead_options = sorted(set(bead_options))
        
        selected_bead = st.sidebar.selectbox("Select Bead Number", bead_options)
        
        if "spectrograms" not in st.session_state:
            st.session_state["spectrograms"] = {}

        for file in selected_files:
            bead_key = f"{file}_bead_{selected_bead}"
            if bead_key not in st.session_state["spectrograms"]:
                if selected_bead <= len(st.session_state["metadata"][file]):
                    df = pd.read_csv(file)
                    start, end = st.session_state["metadata"][file][selected_bead - 1]
                    sample_data = df.iloc[start:end, :2].values
                    
                    fs = 10000
                    nperseg = min(1024, len(sample_data) // 4)
                    noverlap = int(0.99 * nperseg)
                    nfft = min(2048, 4 ** int(np.ceil(np.log2(nperseg * 2))))
                    db_scale = 110
                    
                    f, t, Sxx = signal.spectrogram(sample_data[:, 0], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                    Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
                    min_disp_dB = np.max(Sxx_dB) - db_scale
                    Sxx_dB[Sxx_dB < min_disp_dB] = min_disp_dB
                    
                    st.session_state["spectrograms"][bead_key] = {
                        "f": f,
                        "t": t,
                        "Sxx_dB": Sxx_dB - min_disp_dB,
                        "short_name": shorten_file_name(file)
                    }
        
        spectrograms = {key: data for key, data in st.session_state["spectrograms"].items() if key.endswith(f"bead_{selected_bead}")}
        num_plots = len(spectrograms)
        cols = 6
        rows = int(np.ceil(num_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.clear()
            ax.axis('off')
        
        for i, (key, data) in enumerate(spectrograms.items()):
            ax = axes[i]
            img = ax.pcolormesh(data["t"], data["f"], data["Sxx_dB"], shading='gouraud', cmap='jet')
            ax.set_ylim([0, 500])
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"{data['short_name']} (Bead {selected_bead})")
            fig.colorbar(img, ax=ax, aspect=20)
            ax.axis('on')
        
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        st.pyplot(fig)
