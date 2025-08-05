import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Spectrogram VIZ")

# ========== Setup Session State ==========
if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}
if "spectrograms" not in st.session_state:
    st.session_state["spectrograms"] = {}

# ========== Cached Spectrogram Function ==========
@st.cache_data(show_spinner=False)
def compute_spectrogram(data, fs, nperseg, noverlap, nfft, db_scale):
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
    min_disp_dB = np.max(Sxx_dB) - db_scale
    Sxx_dB[Sxx_dB < min_disp_dB] = min_disp_dB
    return f, t, Sxx_dB - min_disp_dB

# ========== Utilities ==========
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

def shorten_file_name(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    parts = name_without_ext.split("_")
    trimmed_part = "_".join(parts[2:])
    return trimmed_part

# ========== Sidebar UI ==========
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        # Load one sample file to get column names
        sample_file = csv_files[0]
        if sample_file not in st.session_state["csv_dataframes"]:
            st.session_state["csv_dataframes"][sample_file] = pd.read_csv(sample_file)
        df_sample = st.session_state["csv_dataframes"][sample_file]
        columns = df_sample.columns.tolist()
        
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        if st.button("Segment Beads"):
            with st.spinner("Segmenting beads..."):
                metadata = {}
                for file in csv_files:
                    if file not in st.session_state["csv_dataframes"]:
                        st.session_state["csv_dataframes"][file] = pd.read_csv(file)
                    df = st.session_state["csv_dataframes"][file]
                    segments = segment_beads(df, filter_column, threshold)
                    metadata[file] = segments
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata
    
    # Clear button for resetting gallery
    if st.button("🧹 Clear All Spectrograms"):
        st.session_state["spectrograms"] = {}
        st.success("Cleared all spectrograms.")

# ========== Main Visualization ==========
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
        
        # Dropdown to choose which column to use for the spectrogram
        column_to_use = st.sidebar.selectbox("Select column for spectrogram", df_sample.columns[:2])
        
        # Generate or retrieve spectrograms
        for file in selected_files:
            bead_key = f"{file}_bead_{selected_bead}_{column_to_use}"
            if bead_key not in st.session_state["spectrograms"]:
                if selected_bead <= len(st.session_state["metadata"][file]):
                    if file not in st.session_state["csv_dataframes"]:
                        st.session_state["csv_dataframes"][file] = pd.read_csv(file)
                    df = st.session_state["csv_dataframes"][file]
                    start, end = st.session_state["metadata"][file][selected_bead - 1]
                    sample_data = df.iloc[start:end][column_to_use].values
                    
                    fs = 10000
                    nperseg = min(1024, len(sample_data) // 4)
                    noverlap = int(0.99 * nperseg)
                    nfft = min(2048, 4 ** int(np.ceil(np.log2(nperseg * 2))))
                    db_scale = 110
                    
                    f, t, Sxx_dB = compute_spectrogram(sample_data, fs, nperseg, noverlap, nfft, db_scale)
                    st.session_state["spectrograms"][bead_key] = {
                        "f": f,
                        "t": t,
                        "Sxx_dB": Sxx_dB,
                        "short_name": shorten_file_name(file),
                        "bead": selected_bead,
                        "column": column_to_use
                    }

        # 🧹 Prune old spectrograms to limit memory use
        MAX_SPECTROGRAMS = 12
        if len(st.session_state["spectrograms"]) > MAX_SPECTROGRAMS:
            keys_to_remove = list(st.session_state["spectrograms"].keys())[:-MAX_SPECTROGRAMS]
            for k in keys_to_remove:
                del st.session_state["spectrograms"][k]

        # Plotting all spectrograms persistently
        spectrograms = st.session_state["spectrograms"]
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
            ax.set_title(f"{data['short_name']} (Bead {data['bead']}, {data['column']})")
            fig.colorbar(img, ax=ax, aspect=20)
            ax.axis('on')
        
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        st.pyplot(fig)
