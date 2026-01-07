# Minimal script to read CSV data and detect onsets using a pretrained model
# By Imre Vida (imre.vida@mail.polimi.it)
SCRIPT_VERSION = "1.0.0-Minimal-CSV"
# Created: 2025-04-16 (Minimal version: 2025-04-24)

import os
import numpy as np
import pandas as pd # Using pandas for easy CSV reading
import scipy.signal
import tensorflow as tf
import time
import sys
import traceback

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("SciPy version:", scipy.__version__)

print("-------------------------------------")
print("Script version:", SCRIPT_VERSION)
print("-------------------------------------")

# INPUT: Directory containing the CSV signal files
DATA_DIR = './data_csv' #  CSV data directory
# OUTPUT: Directory where onset .txt files will be saved
OUTPUT_DIR = './csv_detected_onsets' # Output directory name 
# MODEL: Path to the pre-trained Keras model
MODEL_PATH = './final_onset_model.h5' # Path to the trained model file

# PARAMETERS (Must match the model's training parameters)
TARGET_FS = 125.0         # Hz (Sampling rate the model expects)
SEGMENT_DURATION = 5      # seconds
SEGMENT_LENGTH = int(SEGMENT_DURATION * TARGET_FS) # Samples per segment
PREDICTION_THRESHOLD = 0.5 # Threshold for converting sigmoid output to binary
PYTHON_BATCH_SIZE = 64     # Batch size for prediction (can be tuned)


def standardize_segments(X_segments):
    """Standardize each segment independently."""
    mean = np.mean(X_segments, axis=1, keepdims=True)
    std = np.std(X_segments, axis=1, keepdims=True)
    # Avoid division by zero for flat segments
    std[std < 1e-6] = 1e-6
    return (X_segments - mean) / std

def create_fixed_length_segments(signal, segment_length):
    """Chops signal into fixed-length segments, discarding the remainder."""
    num_samples = len(signal)
    num_segments = num_samples // segment_length
    if num_segments == 0:
        return np.array([]), 0 # Return empty array and 0 segments
    # Truncate signal to the largest multiple of segment_length
    signal_truncated = signal[:num_segments * segment_length]
    # Reshape into (num_segments, segment_length, 1 feature)
    X_segments = signal_truncated.reshape((num_segments, segment_length, 1)).astype(np.float32)
    return X_segments, num_segments

# --- Main Script ---
print("--- Starting Minimal CSV Onset Detection ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"CSV Data input directory: {os.path.abspath(DATA_DIR)}")
print(f"Onset output directory: {os.path.abspath(OUTPUT_DIR)}")

# Load Model
print(f"Loading Keras model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    sys.exit(1)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary(line_length=100)
    print("Model loaded successfully.")
    # Optional: Check if GPU is used by default
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"TensorFlow detected GPU(s): {gpu_devices}")
        # Basic memory growth setting (optional, but often helpful)
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Warning: Could not set memory growth: {e}")
    else:
        print("No GPU detected by TensorFlow, using CPU.")

except Exception as e:
    print(f"ERROR loading Keras model: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# Find CSV Files
print(f"Scanning for CSV files in: {DATA_DIR}")
if not os.path.isdir(DATA_DIR):
    print(f"ERROR: Data directory not found: {DATA_DIR}")
    sys.exit(1)
csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
if not csv_files:
    print(f"ERROR: No .csv files found in {DATA_DIR}")
    sys.exit(1)
print(f"Found {len(csv_files)} CSV files.")

# --- Process Files ---
total_start_time = time.time()
total_inference_time = 0
files_processed = 0
total_onsets_detected = 0
total_signal_duration_seconds = 0

for filename in csv_files:
    file_start_time = time.time()
    base_name = os.path.splitext(filename)[0] # Get filename without extension
    input_filepath = os.path.join(DATA_DIR, filename)
    print(f"\nProcessing file: {filename}...")

    try:
        # Read CSV Data
        # Assume first row might be header, first col time, second col value
        try:
            
            # Use low_memory=False for potentially large files in case of dtype issues
            df = pd.read_csv(input_filepath, header=0, usecols=[0, 1], low_memory=False)
            # Rename for clarity, assuming col 0 is time, col 1 is value
            df.columns = ['time', 'value']
        except ValueError:
            # If header=0 fails (e.g., no header), try without header
            df = pd.read_csv(input_filepath, header=None, usecols=[0, 1], low_memory=False)
            df.columns = ['time', 'value'] # Assign names manually

        time_vector = df['time'].values.astype(np.float64)
        value_vector = df['value'].values.astype(np.float64)

        if len(value_vector) < 2:
             print(f"  Signal too short (< 2 samples). Skipping.")
             continue

        # preprocessing
        # Check for NaNs in value column
        nan_mask = np.isnan(value_vector)
        if np.all(nan_mask):
            print(f"  Signal contains only NaNs. Skipping.")
            continue
        if np.any(nan_mask):
            print(f"  Warning: Signal contains NaNs. Removing NaN entries.")
            time_vector = time_vector[~nan_mask]
            value_vector = value_vector[~nan_mask]
            if len(value_vector) < 2:
                print(f"  Signal too short after NaN removal. Skipping.")
                continue

        
        dt_original = np.median(np.diff(time_vector))
        if dt_original <= 1e-9 or np.isnan(dt_original):
            print(f"  Warning: Invalid time vector (dt={dt_original}). Skipping.")
            continue
        fs_original = 1.0 / dt_original
        print(f"  Detected original FS: {fs_original:.2f} Hz")

        # Resample if necessary
        if abs(fs_original - TARGET_FS) > 1.0: # Allow small tolerance
            print(f"  Resampling from {fs_original:.2f} Hz to {TARGET_FS} Hz...")
            target_num_samples = int(len(value_vector) * TARGET_FS / fs_original)
            if target_num_samples < 2:
                print(f"  Signal too short to resample. Skipping.")
                continue
            try:
                value_vector = scipy.signal.resample(value_vector, target_num_samples)
                print(f"  Resampled signal length: {len(value_vector)}")
            except Exception as e:
                print(f"  Resampling error: {e}. Skipping.")
                continue
        else:
            print(f"  Original FS is close enough to target FS. No resampling needed.")

        # Check length again after potential resampling
        if len(value_vector) < SEGMENT_LENGTH:
            print(f"  Signal shorter than one segment length ({SEGMENT_LENGTH} samples). Skipping.")
            continue

        # Prepare Segments
        signal_segments, num_segments = create_fixed_length_segments(value_vector, SEGMENT_LENGTH)
        if num_segments == 0:
            print(f"  No full segments could be created. Skipping.")
            continue
        print(f"  Created {num_segments} segments.")

        # Standardize segments
        segments_standardized = standardize_segments(signal_segments)

        # Predict Onsets
        try:
            inference_start_time = time.time()  # Start timing inference
            predicted_probs = model.predict(segments_standardized, batch_size=PYTHON_BATCH_SIZE, verbose=0)
            inference_time = time.time() - inference_start_time  # Measure inference time
            #print(f"  Inference time: {inference_time:.4f} seconds")
            total_inference_time += inference_time
        except Exception as e:
            print(f"  Prediction ERROR: {e}\n{traceback.format_exc()}")
            continue # Skip to next file on prediction error

        # Apply threshold and find onset indices
        predicted_binary = (predicted_probs > PREDICTION_THRESHOLD).astype(np.int8)
        onset_segment_indices = np.where(predicted_binary.flatten() == 1)[0]

        num_onsets = len(onset_segment_indices)
        total_onsets_detected += num_onsets

        # Calculate signal duration (using resampled signal length) and add to total
        signal_duration_seconds = len(value_vector) / TARGET_FS
        total_signal_duration_seconds += signal_duration_seconds

        print(f"  Detected {num_onsets} onset segments.")

        # --- 5. Save Results ---
        if num_onsets > 0:
            # Save onset segment indices to a simple text file
            output_filename = f"{base_name}_onset_segments.txt" # Changed extension
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            try:
                # Save as integers, one index per line
                np.savetxt(output_filepath, onset_segment_indices, fmt='%d')
                print(f"  Saved onset segment indices to {output_filename}")
            except Exception as e:
                print(f"  ERROR saving onsets: {e}")
        else:
            print(f"  No onsets detected, skipping save.")

        files_processed += 1
        file_time = time.time() - file_start_time
        print(f"  Finished processing in {file_time:.2f} seconds.")

    # Catch errors during file processing (e.g., reading CSV, processing signal)
    except pd.errors.EmptyDataError:
        print(f"  Skipping empty or invalid CSV file: {filename}")
    except Exception as e_file:
        print(f"!! ERROR processing file {filename}: {e_file}\n{traceback.format_exc()} !!")

# --- Summary ---
total_time = time.time() - total_start_time
print("\n--- Onset Detection Script Finished ---")
print(f"Total CSV files analyzed: {files_processed}/{len(csv_files)}")
print(f"Total onsets detected: {total_onsets_detected}")
if total_signal_duration_seconds > 0:
    onsets_per_second = total_onsets_detected / total_time
    print(f"Total processed signal duration: {total_signal_duration_seconds:.2f} seconds (at {TARGET_FS} Hz)")
    # Note: This rate is "onset segments per second"
    print(f"Onset detection rate: {onsets_per_second:.4f} onsets/second")
    print(f"Actual inference time (file operations excluded): {total_inference_time:.2f} seconds")
    print(f"Algorithm speed: {total_onsets_detected/total_inference_time:.4f} onsets/second")
else:
    print("Total processed signal duration: 0 seconds")
print(f"Total execution time: {total_time:.2f} seconds")
