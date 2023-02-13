import matplotlib.pyplot as plt
import numpy as np
import struct
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
        frames  = np.divide(frames, max_amp)

        return sample_rate, frames


def find_index_of_nearest(array, value):
    return (np.abs(array - value)).argmin()

    
    

# LA# intelligence
lad_freq  = 466
max_freqs = 32

# Read file
lad_filename = "note_guitare_LAd.wav"
sample_rate, frames = read_file(lad_filename)

# Extract frequencies
data      = np.fft.fft(frames)
freqs_raw = np.fft.fftfreq(len(data))
freqs     = freqs_raw * sample_rate
index_lad = np.argmax(abs(data))
print(index_lad)
print(freqs[index_lad])

#nxt = find_next_index(data, index_lad, index_lad)
#print(nxt)
#exit()

if False:
    amplitudes = np.abs(data)
    amplitudes = np.divide(amplitudes, np.amax(amplitudes))
    # phases     = np.angle(data)

    plt.stem(freqs, amplitudes)
    plt.yscale("log")
    plt.xlim(0, 20000)
    plt.show()


# Get amplitudes at harmonics
index_harms = [index_lad * i     for i in range (0, max_freqs - 1)]
harm_freqs  = [freqs[i]          for i in index_harms]
harmonics   = [np.abs(data[i])   for i in index_harms]
phases      = [np.angle(data[i]) for i in index_harms]

if False:
    print(harmonics)
    plt.stem(harm_freqs[1::], harmonics[1::])
    plt.yscale("log")
    plt.xlim(0, lad_freq * max_freqs)
    plt.show()


def create_sound_from_data(harmonics, phases, fundamental, sampleRate, filename, duration_s = 2):
    audio = []
    ts = np.linspace(0, duration_s , sampleRate * duration_s)

    audio = []
    for t in ts:
        total = 0;
        for i in range(len(harmonics)):
            total += harmonics[i] * np.sin(2 * np.pi * fundamental * i * t + phases[i])

        audio.append(total)

    wav = wave.open(filename, "w")
    nchannels = 1
    sampwidth = 2
    nframes   = len(audio)
    wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

    for sample in audio:
        wav.writeframes(struct.pack('h', int(sample)))

    wav.close()
    

create_sound_from_data(harmonics, phases, freqs[index_lad], sample_rate, "new_note.wav", 2)

