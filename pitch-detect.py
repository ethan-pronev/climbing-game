import numpy as np
import sounddevice as sd

def detect_pitch(signal, samplerate):
    """
    Detects pitch from a given audio signal using autocorrelation.
    
    Parameters:
      signal (np.array): The input audio block (1D numpy array).
      samplerate (int): The sampling rate of the audio.
      
    Returns:
      float: Detected pitch in Hz (or 0 if not found).
    """
    # Apply a Hanning window to reduce spectral leakage.
    window = np.hanning(len(signal))
    signal = signal * window

    # Compute the autocorrelation of the windowed signal.
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # take only the second half

    # Define the pitch range (e.g., human voice: 50Hz to 500Hz).
    min_freq = 50   # Hz
    max_freq = 2000  # Hz
    min_lag = int(samplerate / max_freq)
    max_lag = int(samplerate / min_freq)

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
    pitch = samplerate / lag
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
    pitch = detect_pitch(samples, samplerate)
    
    # Print the detected pitch.
    print(f"Detected pitch: {pitch:.2f} Hz")

# Set stream parameters.
samplerate = 44100  # Hz
block_size = 2048  # Number of samples per block

# Open the audio input stream.
try:
    with sd.InputStream(channels=1, samplerate=samplerate, blocksize=block_size,
                        dtype='float32', callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\nStopped listening.")
except Exception as e:
    print("Error:", str(e))
