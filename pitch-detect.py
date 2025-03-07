import numpy as np
import sounddevice as sd
import argparse
import sys



SAMPLERATE = 44100  # Hz
# BLOCK_SIZE = 2048  # Samples per block
BLOCK_SIZE = 8192

# Typical frequency range for human voice
MIN_FREQ = 50   # Hz
MAX_FREQ = 2000 # Hz



def detect_pitch_autocorr(signal):
    """
    Detects pitch from a given audio signal using autocorrelation.
    
    Parameters:
      signal (np.array): The input audio block (1D numpy array).
      
    Returns:
      float: Detected pitch in Hz (or 0 if not found).
    """
    # Apply a Hanning window to reduce spectral leakage.
    window = np.hanning(len(signal))
    signal = signal * window

    # Compute the autocorrelation of the windowed signal.
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # Take only the second half

    min_lag = int(SAMPLERATE / MAX_FREQ)
    max_lag = int(SAMPLERATE / MIN_FREQ)

    # Only consider lags within our expected pitch range.
    if max_lag > len(corr):
        max_lag = len(corr)
    corr_segment = corr[min_lag:max_lag]
    if len(corr_segment) == 0:
        return 0

    # Find the lag with the maximum autocorrelation value.
    peak_index = np.argmax(corr_segment)
    lag = peak_index + min_lag
    if lag == 0:
        return 0

    # Calculate the pitch from the lag.
    pitch = SAMPLERATE / lag
    return pitch

def detect_pitch_fft(signal):
    """
    Detects pitch from a given audio signal using FFT.

    Parameters:
      signal (np.array): The input audio block (1D numpy array).

    Returns:
      float: Detected pitch in Hz (or 0 if not found).
    """
    # Apply a Hanning window to reduce spectral leakage.
    window = np.hanning(len(signal))
    signal = signal * window

    # Compute FFT and get the magnitude spectrum.
    fft_spectrum = np.fft.rfft(signal)  # Real FFT (faster for real signals)
    magnitudes = np.abs(fft_spectrum)   # Get magnitude of frequencies

    # Get the frequency values corresponding to FFT bins.
    freqs = np.fft.rfftfreq(len(signal), d=1/SAMPLERATE)

    # Filter indices that correspond to valid frequencies.
    valid_indices = np.where((freqs >= MIN_FREQ) & (freqs <= MAX_FREQ))[0]

    if len(valid_indices) == 0:
        return 0  # No valid frequencies detected

    # Find the peak frequency.
    peak_index = valid_indices[np.argmax(magnitudes[valid_indices])]
    pitch = freqs[peak_index]
    return pitch

def audio_callback(indata, frames, time, status):
    """
    This callback processes each audio block to detect pitch.
    """
    if status:
        print("Stream status:", status)

    # Assume mono input; if stereo, pick one channel.
    samples = indata[:, 0]

    # Compute the pitch for the current block.
    pitch = pitch_detection_function(samples)  # Use selected method

    # Print the detected pitch.
    print(f"\rDetected pitch: {pitch:.2f} Hz     ", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live pitch detection using Autocorrelation or FFT.")
    parser.add_argument("--method", choices=["autocorr", "fft"], required=True,
                        help="Choose pitch detection method: 'autocorr' or 'fft' (required)")
    args = parser.parse_args()

    # Assign the selected pitch detection function
    if args.method == "fft":
        pitch_detection_function = detect_pitch_fft
    elif args.method == "autocorr":
        pitch_detection_function = detect_pitch_autocorr
    else:
        print("Error: Invalid method. Use '--method autocorr' or '--method fft'.")
        sys.exit(1)

    # Open the audio input stream.
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLERATE, blocksize=BLOCK_SIZE,
                            dtype='float32', callback=audio_callback):
            print(f"Listening using {args.method.upper()} method... Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped listening.")
    except Exception as e:
        print("Error:", str(e))
