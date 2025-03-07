import numpy as np
import sounddevice as sd

# Audio stream parameters
samplerate = 44100  # in Hz
block_size = 2048   # number of samples per block

# Detection thresholds (tune these based on your environment)
energy_threshold = 0.02         # Minimum RMS energy to consider an event
spectral_centroid_threshold = 4000  # Frequency (Hz) threshold to distinguish snap vs. clap
transient_factor = 2.0          # Require the block energy to be at least 3x the recent average
voice_threshold = 2000

# To store energy of previous blocks to help decide if an event is transient
energy_history = []
history_length = 5  # number of previous blocks to average

def compute_rms(signal):
    """Compute the root mean square energy of the signal."""
    return np.sqrt(np.mean(signal**2))

def compute_spectral_centroid(signal, samplerate):
    """
    Compute the spectral centroid of a signal.
    The spectral centroid indicates the "brightness" of a sound.
    """
    # Apply a Hanning window to reduce spectral leakage.
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    # Compute the FFT magnitude spectrum
    fft_vals = np.abs(np.fft.rfft(windowed_signal))
    # Frequency bins corresponding to the FFT
    freqs = np.fft.rfftfreq(len(signal), d=1.0/samplerate)
    if np.sum(fft_vals) == 0:
        return 0
    centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    return centroid

def audio_callback(indata, frames, time, status):
    """
    This callback processes each audio block to detect claps and snaps,
    while filtering out continuous sounds such as regular speech.
    """
    global energy_history

    if status:
        print("Stream status:", status)
    
    # Use only the first channel (mono)
    signal = indata[:, 0]
    
    # Compute the RMS energy of the current block
    rms = compute_rms(signal)
    
    # Update the energy history and compute a moving average
    energy_history.append(rms)
    if len(energy_history) > history_length:
        energy_history.pop(0)
    avg_energy = np.mean(energy_history) if energy_history else energy_threshold

    # Only consider this block if it's a sudden (transient) spike in energy.
    # This helps avoid detecting continuous sounds (like speech).
    if rms > energy_threshold and rms > transient_factor * avg_energy:
        centroid = compute_spectral_centroid(signal, samplerate)
        
        # Decide if the event is a snap (higher spectral centroid) or a clap.
        if centroid < voice_threshold:
            pass
        elif centroid > spectral_centroid_threshold:
            print(f"Detected SNAP! RMS: {rms:.4f}, Centroid: {centroid:.2f} Hz")
        else:
            print(f"Detected CLAP! RMS: {rms:.4f}, Centroid: {centroid:.2f} Hz")
    # Otherwise, ignore the event (likely part of continuous speech)
    else:
        pass

try:
    with sd.InputStream(channels=1, samplerate=samplerate, blocksize=block_size,
                        dtype='float32', callback=audio_callback):
        print("Listening for claps and snaps (excluding regular speech). Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print("Error:", e)