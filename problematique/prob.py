import matplotlib.pyplot as plt
import numpy as np
import struct
import wave
from scipy import signal

def main():
    # La#
    lad_freq       = 466
    harmonicsCount = 32

    # Read file
    lad_filename = "note_guitare_LAd.wav"
    sampleRate, frames = read_file(lad_filename)

    # Get enveloppe using lowpass filter
    N = get_filter_order(np.pi/1000, sampleRate)
    enveloppe = apply_lowpass(N, frames)
    enveloppe = np.divide(enveloppe, np.amax(enveloppe))    # Normalizing enveloppe
    
    # Apply Hamming window
    frames = hamming(frames)

    # Get information from frames
    harmonics, phases, fundamental = extract_frequencies(frames, sampleRate, harmonicsCount)

    # Get note frequencies
    note_freqs = generate_note_frequencies(fundamental)     # Generate all the other notes from LA#

    if True:
        synthesize_lad(harmonics, phases, fundamental, sampleRate, enveloppe)

    if True:
        synthesize_beethoven(harmonics, phases, sampleRate, enveloppe, note_freqs)



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

def get_filter_order(omega, sampleRate):
    #filtre passebas
    
    #f = (𝜔̅ /2 )Fe 
    fe = sampleRate
    fc = (omega *fe )/ (2 * np.pi)
    #|H(𝜔̅)| = 1/N * ∑ e^(-j𝜔̅n)
    gain = np.power(10, -3/20)
    #gain = np.sqrt(2)/2
    
    err = []
    H0 = 1
    hGain = []
    for M in range(1,1000,1):
        sum = 0
        for i in range(0,M,1):
            sum += (np.exp(-1j * 0 * i))
        a = H0/sum.real
        currentGain =  0
        for k in range(0,M,1):
            currentGain += (np.exp(-1j * omega * k))
        hGain.append(np.abs(a * currentGain))
    N = find_index_of_nearest(hGain, gain) +1
    print("Lowpass Filter Order: " + str(N))
    return N

def apply_lowpass(N, frames):
    lowPass = [1/N for n in range(N)]
	
    return np.convolve(lowPass, np.abs(frames))


def pad_thai(array, length):
    return np.pad(array, (0, length - len(array)))

def unpad_thai(array, length):
    return array[0:length]


def find_index_of_nearest(array, value):
    return (np.abs(array - value)).argmin()

def hamming(frames):
    hamming = np.hamming(len(frames))
    return np.multiply(frames, hamming)


def display_frequencies(harmonics, phases, fftData, frames):
	fig, (frams, fft, harm, phas)  = plt.subplots(4)
	
	frams.plot(frames)
	#frams.set_xlim(0, 20000)
	fft.stem(fftData)
	fft.set_xlim(0, 20000)
	fft.set_yscale("log")
	harm.stem(harmonics)
	harm.set_yscale("log")
	phas.stem(phases)
	
	plt.show()

def extract_frequencies(frames, sampleRate, harmonicsCount):
    # Extract frequencies
    data      = np.fft.fft(frames)
    freqs_raw = np.fft.fftfreq(len(data))
    freqs     = freqs_raw * sampleRate
	
    index_lad = np.argmax(abs(data))
    fundamental = freqs[index_lad]
    #print(index_lad)
    print("La# fundamental frequency: " + str(fundamental))
	

    # Get amplitudes at harmonics
    index_harms = [index_lad * i     for i in range (0, harmonicsCount + 1)]
    harm_freqs  = [freqs[i]          for i in index_harms]
    harmonics   = [np.abs(data[i])   for i in index_harms]
    phases      = [np.angle(data[i]) for i in index_harms]

    # display_frequencies(harmonics, phases, data, frames)
	
    return harmonics, phases, fundamental
    

def create_audio(harmonics, phases, fundamental, sampleRate, enveloppe, duration_s = 2):
    audio = []
    ts = np.linspace(0, duration_s , int(sampleRate * duration_s))

    audio = []
    for t in ts:
        total = 0;
        for i in range(len(harmonics)):
            total += 10 * harmonics[i] * np.sin(2 * np.pi * fundamental * i * t + phases[i])

        audio.append(total)

    # Apply enveloppe
    new_env   = unpad_thai(enveloppe, len(audio))
    new_audio = pad_thai(audio, len(new_env))
    audio     = np.multiply(audio, new_env)

    # Apply second window
    # Apply Hamming window
    # audio = hamming(audio)
    return audio.tolist()

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

def generate_note_frequencies(lad_freq):
    la_freq = lad_freq / 1.06

    frequencies = {
        "do"   : la_freq * 0.595,
        "do#"  : la_freq * 0.630,
        "ré"   : la_freq * 0.667,
        "ré#"  : la_freq * 0.707,
        "mi"   : la_freq * 0.749,
        "fa"   : la_freq * 0.794,
        "fa#"  : la_freq * 0.841,
        "sol"  : la_freq * 0.891,
        "sol#" : la_freq * 0.944,
        "la"   : la_freq,
        "la#"  : lad_freq,
        "si"   : la_freq * 1.123
    }

    return frequencies

def synthesize_lad(harmonics, phases, fundamental, sampleRate, enveloppe):
    lad_audio = create_audio(harmonics, phases, fundamental, sampleRate, enveloppe, 2)
	
    # extract_frequencies(lad_audio, sampleRate, 32)
	
    create_wav_from_audio(lad_audio, sampleRate, "LA#.wav")

def synthesize_beethoven(harmonics, phases, sampleRate, enveloppe, note_freqs):
    sol_audio  = create_audio(harmonics, phases, note_freqs["sol"], sampleRate, enveloppe, 0.4)
    mib_audio  = create_audio(harmonics, phases, note_freqs["ré#"], sampleRate, enveloppe, 1.5)
    fa_audio   = create_audio(harmonics, phases, note_freqs["fa"],  sampleRate, enveloppe, 0.4)
    re_audio   = create_audio(harmonics, phases, note_freqs["ré"],  sampleRate, enveloppe, 1.5)
    silence_1  = create_silence(sampleRate, 0.2)
    silence_2  = create_silence(sampleRate, 1.5)

    beethoven = sol_audio + silence_1 + \
                sol_audio + silence_1 + \
                sol_audio + silence_1 + \
                mib_audio + silence_2 + \
                fa_audio  + silence_1 + \
                fa_audio  + silence_1 + \
                fa_audio  + silence_1 + \
                re_audio

    create_wav_from_audio(beethoven, sampleRate, "beethoven.wav")



if __name__ == "__main__":
    main()
