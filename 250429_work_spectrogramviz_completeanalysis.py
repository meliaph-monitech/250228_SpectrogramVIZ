import streamlit as st
import zipfile
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.signal import spectrogram, welch, find_peaks
from sklearn.decomposition import PCA
from tempfile import TemporaryDirectory

# --- Streamlit Config ---
st.set_page_config(layout="wide")
st.title("🎧 Welding Sound Analyzer V3 (WAV + CSV Modes)")

# --- Utility Functions ---
def extract_label(filename):
    return filename.split("_")[-1].replace(".wav", "").upper()

def is_wav_zip(zip_file):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        return any(f.endswith(".wav") for f in zip_ref.namelist())

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

# --- File Upload ---
uploaded_zip = st.file_uploader("Upload a ZIP file containing WAV or CSV files", type="zip")

label_colors = {
    "OK": "green",
    "GAP": "orange",
    "POWER": "red"
}

# --- Main Logic ---
if uploaded_zip:

    # Save temporarily
    with open("temp_uploaded.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())

    if is_wav_zip("temp_uploaded.zip"):
        # --- WAV Mode ---
        with zipfile.ZipFile("temp_uploaded.zip", "r") as zip_ref:
            wav_files = [f for f in zip_ref.namelist() if f.endswith(".wav")]
            if not wav_files:
                st.warning("No .wav files found.")
            else:
                selected_files = st.multiselect("Select WAV files to analyze", wav_files)
                available_labels = sorted(set(extract_label(f) for f in selected_files))
                selected_labels = st.multiselect("Filter by Label", options=available_labels, default=available_labels)

                interval_ms = st.slider("Sampling interval (ms)", 1, 100, 10)

                st.markdown("### Frequency Visualization Settings")
                col1, col2, col3, col4 = st.columns(4)
                min_freq = col1.number_input("Min Frequency (Hz)", value=0)
                max_freq = col2.number_input("Max Frequency (Hz)", value=20000)
                min_db = col3.number_input("Min dB", value=-100)
                max_db = col4.number_input("Max dB", value=0)

                waveform_fig = go.Figure()
                freq_fig = go.Figure()
                peak_data = []
                band_energy_data = []
                pca_vectors = []
                labels = []
                file_labels = []
                processed_csvs = {}

                for file in selected_files:
                    label = extract_label(file)
                    if label not in selected_labels:
                        continue

                    color = label_colors.get(label, "gray")

                    with zip_ref.open(file) as wav_file:
                        samplerate, data = wavfile.read(io.BytesIO(wav_file.read()))
                        if data.ndim > 1:
                            data = data.mean(axis=1)

                        # Time-domain
                        window_size = int(samplerate * interval_ms / 1000)
                        trimmed = len(data) - len(data) % window_size
                        reshaped = data[:trimmed].reshape(-1, window_size)
                        avg = reshaped.mean(axis=1)
                        time_axis = np.arange(len(avg)) * interval_ms

                        df = pd.DataFrame({"Time (ms)": time_axis, "Amplitude": avg})
                        processed_csvs[file] = df

                        waveform_fig.add_trace(go.Scatter(x=df["Time (ms)"], y=df["Amplitude"],
                                                          mode="lines", name=f"{file} ({label})",
                                                          line=dict(color=color)))

                        # FFT
                        freqs, psd = welch(data, fs=samplerate, nperseg=2048)
                        db = 10 * np.log10(psd + 1e-12)
                        mask = (freqs >= min_freq) & (freqs <= max_freq)

                        freq_fig.add_trace(go.Scatter(
                            x=freqs[mask], y=np.clip(db[mask], min_db, max_db),
                            mode="lines", fill='tozeroy', name=f"{file} ({label})",
                            line=dict(color=color)
                        ))

                        # Band Energy
                        bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
                        energy_per_band = []
                        for b_start, b_end in bands:
                            band_mask = (freqs >= b_start) & (freqs < b_end)
                            energy = np.mean(psd[band_mask]) if band_mask.any() else 0
                            energy_per_band.append(energy)
                        band_energy_data.append((file, label, energy_per_band))

                        # Peak Frequencies
                        peaks, _ = find_peaks(db, height=np.max(db) - 10)
                        peak_freqs = freqs[peaks]
                        peak_data.extend(peak_freqs)

                        # PCA
                        fft_profile = db[mask]
                        pca_vectors.append(fft_profile)
                        labels.append(label)
                        file_labels.append(file)

                if selected_files and selected_labels:
                    st.subheader("📈 Time-Domain Signal")
                    st.plotly_chart(waveform_fig, use_container_width=True)

                    st.subheader("🔊 Frequency-Domain Spectrum")
                    freq_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="dB")
                    st.plotly_chart(freq_fig, use_container_width=True)

                    # Band Energy Bar
                    st.subheader("📊 Band Energy (Bar & Radar)")
                    band_labels = [f"{start//1000}-{end//1000}kHz" for start, end in bands]

                    bar_fig = go.Figure()
                    for file, label, energies in band_energy_data:
                        if label in selected_labels:
                            color = label_colors.get(label, "gray")
                            bar_fig.add_trace(go.Bar(x=band_labels, y=energies, name=f"{file} ({label})", marker_color=color))
                    bar_fig.update_layout(barmode="group", xaxis_title="Frequency Band", yaxis_title="Avg Energy")
                    st.plotly_chart(bar_fig, use_container_width=True)

                    radar_fig = go.Figure()
                    for file, label, energies in band_energy_data:
                        if label in selected_labels:
                            color = label_colors.get(label, "gray")
                            radar_fig.add_trace(go.Scatterpolar(
                                r=energies + [energies[0]],
                                theta=band_labels + [band_labels[0]],
                                fill='toself',
                                name=f"{file} ({label})",
                                line=dict(color=color)
                            ))
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                    st.plotly_chart(radar_fig, use_container_width=True)

                    # Peak Histogram
                    st.subheader("📌 Histogram of FFT Peak Frequencies")
                    hist_fig = go.Figure()
                    hist_fig.add_trace(go.Histogram(x=peak_data, nbinsx=50))
                    hist_fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Count")
                    st.plotly_chart(hist_fig, use_container_width=True)

                    # PCA Plot
                    st.subheader("📉 PCA - Frequency Profile Projection")
                    try:
                        pca = PCA(n_components=2)
                        reduced = pca.fit_transform(np.array(pca_vectors))
                        pca_fig = go.Figure()
                        for i, label in enumerate(labels):
                            if label in selected_labels:
                                color = label_colors.get(label, "gray")
                                pca_fig.add_trace(go.Scatter(
                                    x=[reduced[i, 0]], y=[reduced[i, 1]],
                                    mode="markers+text", text=[file_labels[i]], name=label,
                                    textposition="top center", marker=dict(color=color, size=10)
                                ))
                        pca_fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
                        st.plotly_chart(pca_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"PCA Error: {e}")

    else:
        # --- CSV Mode ---
        csv_files, extract_dir = extract_zip("temp_uploaded.zip")
        st.success(f"Extracted {len(csv_files)} CSV files.")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()

        with st.sidebar:
            filter_column = st.selectbox("Select column for filtering", columns)
            threshold = st.number_input("Enter filtering threshold", value=0.0)
            if st.button("Segment Beads"):
                metadata = {}
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    metadata[file] = segments
                st.session_state["metadata"] = metadata
                st.success("Bead segmentation complete.")

            fs = st.number_input("Sampling Frequency (fs)", min_value=1000, max_value=50000, value=10000)
            nperseg = st.number_input("nperseg parameter", min_value=256, max_value=4096, value=1024)
            noverlap_factor = st.slider("Overlap Ratio", min_value=0.0, max_value=1.0, value=0.5)
            nfft = st.number_input("nfft parameter", min_value=512, max_value=8192, value=2048)

        if "metadata" in st.session_state and isinstance(st.session_state["metadata"], dict):
            selected_files = st.sidebar.multiselect("Select CSV files", list(st.session_state["metadata"].keys()))
            if selected_files:
                selected_bead = st.sidebar.number_input("Select Bead Number", min_value=1, value=1)
                column_options = df_sample.columns.tolist()
                selected_column = st.sidebar.selectbox("Select Data Column", column_options)
                frequency = st.sidebar.number_input("Enter Frequency (Hz)", min_value=1, value=240)

                fig = go.Figure()
                waveform_fig = go.Figure()
                freq_fig = go.Figure()
                peak_data = []
                band_energy_data = []
                pca_vectors = []
                labels = []
                file_labels = []

                for selected_file in selected_files:
                    df = pd.read_csv(selected_file)
                    if selected_bead <= len(st.session_state["metadata"][selected_file]):
                        start, end = st.session_state["metadata"][selected_file][selected_bead - 1]
                        sample_data = df.iloc[start:end, :]

                        # Spectrogram
                        noverlap = int(noverlap_factor * nperseg)
                        f, t, Sxx = spectrogram(
                            sample_data[selected_column].to_numpy(),
                            fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft
                        )
                        Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

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

                        # --- Additional Figures ---
                        # FFT
                        freqs, psd = welch(sample_data[selected_column], fs=fs, nperseg=nperseg)
                        db = 10 * np.log10(psd + 1e-12)
                        mask = (freqs >= 0) & (freqs <= 20000)

                        freq_fig.add_trace(go.Scatter(
                            x=freqs[mask], y=db[mask],
                            mode="lines", name=f"{selected_file}"
                        ))

                        # Band Energy
                        bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
                        energy_per_band = []
                        for b_start, b_end in bands:
                            band_mask = (freqs >= b_start) & (freqs < b_end)
                            energy = np.mean(psd[band_mask]) if band_mask.any() else 0
                            energy_per_band.append(energy)
                        band_energy_data.append((selected_file, "CSV", energy_per_band))

                        # Peak Frequencies
                        peaks, _ = find_peaks(db, height=np.max(db) - 10)
                        peak_freqs = freqs[peaks]
                        peak_data.extend(peak_freqs)

                        # PCA
                        pca_vectors.append(db[mask])
                        labels.append("CSV")
                        file_labels.append(selected_file)

                # --- Plots ---
                st.subheader("Signal Intensity at Selected Frequency")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("FFT of Beads")
                st.plotly_chart(freq_fig, use_container_width=True)

                st.subheader("Peak Frequencies Histogram")
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Histogram(x=peak_data, nbinsx=50))
                st.plotly_chart(hist_fig, use_container_width=True)

                st.subheader("Band Energy Plot")
                band_labels = [f"{start//1000}-{end//1000}kHz" for start, end in bands]
                bar_fig = go.Figure()
                for file, label, energies in band_energy_data:
                    bar_fig.add_trace(go.Bar(x=band_labels, y=energies, name=file))
                st.plotly_chart(bar_fig, use_container_width=True)

                st.subheader("PCA Projection")
                try:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(np.array(pca_vectors))
                    pca_fig = go.Figure()
                    for i in range(len(file_labels)):
                        pca_fig.add_trace(go.Scatter(
                            x=[reduced[i, 0]], y=[reduced[i, 1]],
                            mode="markers+text", text=[file_labels[i]],
                            textposition="top center"
                        ))
                    st.plotly_chart(pca_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"PCA Error: {e}")

