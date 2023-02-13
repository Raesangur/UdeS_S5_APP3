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

# Apply Hamming window
hamming = np.hanning(len(frames))
frames  = np.multiply(frames, hamming)

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


def create_audio(harmonics, phases, fundamental, sampleRate, duration_s = 2):
    audio = []
    ts = np.linspace(0, duration_s , int(sampleRate * duration_s))

    audio = []
    for t in ts:
        total = 0;
        for i in range(len(harmonics)):
            total += harmonics[i] * np.sin(2 * np.pi * fundamental * i * t + phases[i])

        audio.append(total)

    return audio

def create_silence(sampleRate, duration_s = 1):
    return [0 for t in np.linspace(0, duration_s , int(sampleRate * duration_s))]


def create_wav_from_audio(audio, sampleRate, filename):
    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes   = len(audio)
        wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

        for sample in audio:
            wav.writeframes(struct.pack('h', int(sample)))
    
if False:
    lad_audio = create_audio(harmonics, phases, freqs[index_lad], sample_rate, 2)
    create_wav_from_audio(lad_audio, sample_rate, "LA#.wav")


# Create Beethoven
sol_freq = 392.0
mi_freq  = 329.6
fa_freq  = 349.2
re_freq  = 293.7


sol_audio = create_audio(harmonics, phases, sol_freq, sample_rate, 0.4)
silence_1 = create_silence(sample_rate, 0.2)
mi_audio  = create_audio(harmonics, phases, mi_freq,  sample_rate, 1.5)
silence_2 = create_silence(sample_rate, 1.5)
fa_audio  = create_audio(harmonics, phases, fa_freq,  sample_rate, 0.4)
re_audio = create_audio(harmonics, phases, re_freq,  sample_rate, 1.5)

beethoven = sol_audio + silence_1 + \
            sol_audio + silence_1 + \
            sol_audio + silence_1 + \
            mi_audio  + silence_2 + \
            fa_audio  + silence_1 + \
            fa_audio  + silence_1 + \
            fa_audio  + silence_1 + \
            re_audio

create_wav_from_audio(beethoven, sample_rate, "beethoven.wav")

