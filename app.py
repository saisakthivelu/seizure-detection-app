import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

WINDOW_SIZE = 178

# ------------------ Functions ------------------

def preprocess_eeg(file):
    df = pd.read_excel(file)
    df = df.dropna()
    signal = df.mean(axis=1).values
    return signal

def segment_signal(signal):
    segments = []
    for i in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        segment = signal[i:i + WINDOW_SIZE]
        segments.append(segment)
    return np.array(segments)

# ------------------ Load Model ------------------

model = load_model("seizure_detection_model.h5", compile=False)

# ------------------ UI ------------------

st.title("EEG Seizure Detection System")

uploaded_file = st.file_uploader("Upload EEG File (.xlsx)", type=["xlsx"])

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    # Preprocess
    signal = preprocess_eeg(uploaded_file)
    segments = segment_signal(signal)

    segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

    # Prediction
    predictions = model.predict(segments)
    predictions = (predictions > 0.5).astype(int)

    seizure_count = int(np.sum(predictions))
    normal_count = int(len(predictions) - seizure_count)

    seizure_ratio = round(seizure_count / len(predictions), 2)

    # Final result
    if seizure_ratio > 0.3:
        result = "SEIZURE DETECTED"
        st.error(result)
    else:
        result = "NORMAL EEG"
        st.success(result)

    # Show details
    st.write("### Results")
    st.write("Seizure Segments:", seizure_count)
    st.write("Normal Segments:", normal_count)
    st.write("Seizure Ratio:", seizure_ratio)

    # ------------------ Graph ------------------

    fig, ax = plt.subplots(2, 1, figsize=(10,6))

    # Raw EEG
    ax[0].plot(signal)
    ax[0].set_title("Raw EEG Signal")

    # Segments
    start = 0
    for i, seg in enumerate(segments[:,:,0]):
        end = start + WINDOW_SIZE
        if predictions[i] == 1:
            ax[1].plot(range(start, end), seg,color='red')
        else:
            ax[1].plot(range(start, end), seg,color='blue')
        start = end

    ax[1].set_title("Segment Classification")

    st.pyplot(fig)