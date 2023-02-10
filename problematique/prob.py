import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy import signal


# Reads the .wav file
# Returns the sample_rate in Hz
# Returns an array of samples, with amplitudes normalized at 1
def read_file(filename):
    with wave.open(filename, "rb") as wav:
        # Extract info from wave file
        sample_rate = wav.getframerate()
        frames      = wav.readframes(-1)
        frames      = np.frombuffer(frames, dtype=np.int16)

        # Normalize at 1
        max_amp = np.amax(frames)
        frames = np.divide(frames, max_amp)

        return sample_rate, frames


def find_index_of_nearest(array, value):
    return (np.abs(array - value)).argmin()

# LA# intelligence
lad_freq     = 466
max_freqs    = 32

# Read file
lad_filename = "note_guitare_LAd.wav"
sample_rate, frames = read_file(lad_filename)

# Extract frequencies
data  = np.fft.fft(frames)
freqs_raw = np.fft.fftfreq(len(data))
freqs = freqs_raw * sample_rate
index_lad = np.argmax(abs(data))
print(index_lad)
print(freqs[index_lad])

if False:
    amplitudes = np.abs(data)
    amplitudes = np.divide(amplitudes, np.amax(amplitudes))
    phases     = np.angle(data)

    plt.stem(freqs, amplitudes)
    plt.yscale("log")
    plt.xlim(0, 20000)
    plt.show()

# Get amplitudes at harmonics
harmonics = [data[index_lad * i] for i in range(0, 31)]
print(harmonics)

# Printing LA#
print(sample_rate)
fig, axs = plt.subplots(3)
axs[0].plot(frames)
axs[1].plot(np.abs(data))
axs[1].set_xlim(0, lad_freq * max_freqs)
axs[2].stem(np.abs(harmonics))
plt.show()
