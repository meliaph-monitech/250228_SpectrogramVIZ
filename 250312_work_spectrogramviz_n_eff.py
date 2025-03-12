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
        # Check if the directory contains any files before attempting to remove them
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path):  # Ensure it's a file (not a subdirectory)
                os.remove(file_path)
    else:
        os.makedirs(extract_dir)  # Create the directory if it doesn't exist
    
    # Extract the ZIP file contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Collect CSV files from the extracted directory
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
    """
    Extracts the hhmmss and nn parts of the file name from the format:
    extracted_csvs/YYMMDD_hhmmss_*_nn.csv
    """
    base_name = os.path.basename(file_path)
    parts = base_name.split("_")
    hhmmss = parts[1]  # The second part is always the time (hhmmss)
    nn = parts[-1].split(".")[0]  # The last part before .csv is the nn
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
    # Display shortened file names in the sidebar
    shortened_file_names = {shorten_file_name(file): file for file in st.session_state["metadata"].keys()}
    
    # Sort the keys (shortened file names) alphabetically
    sorted_shortened_names = sorted(shortened_file_names.keys())
    
    # Use the sorted list in the multiselect dropdown
    selected_files_short = st.sidebar.multiselect("Select CSV files", sorted_shortened_names, key="selected_files")
    
    # Map back to full file paths
    selected_files = [shortened_file_names[short_name] for short_name in selected_files_short]
    
    selected_bead = st.sidebar.number_input("Select Bead Number", min_value=1, value=1, step=1)
    
    # Initialize session state for spectrogram data
    if "spectrograms" not in st.session_state:
        st.session_state["spectrograms"] = {}

    # Update session state based on selected files
    for file in selected_files:
        if file not in st.session_state["spectrograms"]:
            # Compute spectrogram only for newly selected files
            if selected_bead <= len(st.session_state["metadata"][file]):  # Ensure bead number is valid
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
                
                # Store the spectrogram data in session state
                st.session_state["spectrograms"][file] = {
                    "f": f,
                    "t": t,
                    "Sxx_dB": Sxx_dB - min_disp_dB,
                    "short_name": shorten_file_name(file)
                }
    
    # Remove spectrograms for deselected files
    for file in list(st.session_state["spectrograms"].keys()):
        if file not in selected_files:
            del st.session_state["spectrograms"][file]

    # Render the spectrograms
    spectrograms = st.session_state["spectrograms"]
    num_plots = len(spectrograms)
    cols = 6  # Fixed number of columns per row
    rows = int(np.ceil(max(len(shortened_file_names), 1) / cols))  # Calculate rows dynamically based on the maximum number of files

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten axes array to handle cases with fewer plots

    # Clear all axes to ensure unused subplots are empty
    for ax in axes:
        ax.clear()
        ax.axis('off')  # Hide unused axes initially

    # Populate the axes with selected spectrograms
    for i, (file, data) in enumerate(spectrograms.items()):
        ax = axes[i]
        img = ax.pcolormesh(data["t"], data["f"], data["Sxx_dB"], shading='gouraud', cmap='jet')
        ax.set_ylim([0, 500])
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Bead {selected_bead}\n{data['short_name']}")
        fig.colorbar(img, ax=ax, aspect=20)
        ax.axis('on')  # Turn on the axis for active plots

    # Adjust spacing between rows and columns to prevent overlap
    plt.subplots_adjust(hspace=0.65, wspace=0.4)

    # Render the figure
    st.pyplot(fig)
