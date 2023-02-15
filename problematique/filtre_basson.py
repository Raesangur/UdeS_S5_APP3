import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import wave as wave
from scipy import signal
import struct

#read file and return sample rate and frames
def read_wav(filename):
    with wave.open(filename) as wav:   # Ouverture du fichier audio
        sample_rate = wav.getframerate()
        frames      = wav.readframes(-1)
        frames      = np.frombuffer(frames, dtype=np.int16)
    return sample_rate, frames
#create a wav file from audio
def create_wav_from_audio(audio, sampleRate, filename):
    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes   = len(audio)
        wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

        for sample in audio:
            wav.writeframes(struct.pack('h', np.int16(sample)))
            
#pour faire le filtre en cascade repasser le fichier créer dans le read pour qu'il soit refiltrer 

#D:/Documents/uni/S5/APP3/note_basson_plus_sinus_1000_Hz_filtre_modifier.wav (fichier créer)
sample_rate, frames = read_wav("D:/Documents/uni/S5/APP3/note_basson_plus_sinus_1000_Hz.wav")
#filtre coupe-bande
N2=6000
fe2 = sample_rate
fc1 = 1000
fc2 = 40
w0 = (2 * np.pi * fc1) / fe2
w1 = (2 * np.pi * fc2) / fe2
m2 = (fc2 * N2) / fe2
k2 = (2 * m2) +1
data = np.linspace(-(N2/2)+1, N2/2, N2)
dn=[1 if data[i] == 0 else 0 for i in range(0,N2,1)]
hLp = []
for el in data:
      if el == 0:
        dn.append(1)
        hLp.append(k2/N2)
      else:
        dn.append(0)
        hLp.append((np.sin((np.pi * el * k2) / N2) / np.sin((np.pi * el) / N2))/N2)

plt.plot(data,np.abs(hLp))
plt.title("Filtre passe-bas avec la technique de la fenêtrage 40 hz pour obtenir un filtre coupe-bande")
plt.xlabel("Échantillon")
plt.ylabel("Amplitude")
plt.show()

hnBasson =[dn[i] - np.multiply(2 * hLp[i] , np.cos(w0 * data[i])) for i in range(0,N2,1)]

plt.plot(data,hnBasson)
plt.title("Réponse impulsionnelle du filtre coupe-bande")
#plt.xlim(-100,100)
plt.xlabel("Échantillon")
plt.ylabel("Amplitude")
plt.show()

sin = np.sin(1000 * 2 * np.pi)
resultSin = np.convolve(sin, hnBasson)

plt.plot(resultSin)
plt.xlabel("Échantillon")
plt.ylabel("Amplitude")
plt.show()

HnBasson = np.fft.fft(hnBasson)
cb_freqs = np.fft.fftfreq(len(hnBasson),d=1/fe2)

plt.plot(cb_freqs[:500],20*np.log10(np.abs(HnBasson[:500])))
plt.title("Visualisation de la fréquence de coupure")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

w, amp = signal.freqz(hnBasson)
angles = np.unwrap(np.angle(amp))
fig, ax1 = plt.subplots()
ax1.set_title('Réponse en fréquence du filtre coupe-bande')
ax1.plot(w,20*np.log10(np.abs(amp)))
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Fréquence normalisé [rad/échantillon]')
ax2 = ax1.twinx()
ax2.plot(w, angles, 'g-')
ax2.set_ylabel('Phase (rad)', color='g')
ax2.grid(True)
ax2.axis('tight')
plt.show()

fftBassonOrig = np.fft.fft(frames)
plt.plot(20*np.log10(np.abs(fftBassonOrig)))
plt.title("Spectres de fourier avant filtrage")
plt.xlabel("fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

#Convolution du signal basson avec le filtre
fftBassonOrig = np.fft.fft(frames)
bassonOri_freqs = np.fft.fftfreq(len(frames),d=1/sample_rate)

plt.plot(bassonOri_freqs,20*np.log10(np.abs(fftBassonOrig)))
plt.xlabel("fréquence (Hz)")
plt.xlim(0,1500)
plt.ylabel("Amplitude (dB)")
plt.show()

result = np.convolve(frames, hnBasson)
plt.plot(result)
plt.xlabel("Échantillons")
plt.ylabel("Amplitude")
plt.show()

fftBassonFiltre = np.fft.fft(result)
bassonFin_freqs = np.fft.fftfreq(len(result),d=1/sample_rate)
plt.plot(bassonFin_freqs,20*np.log10(np.abs(fftBassonFiltre)))
plt.xlim(0,1500)
plt.xlabel("fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

fenetre = np.hanning(len(result))*result
plt.plot(fenetre)
plt.xlabel("Échantillons")
plt.ylabel("Amplitude")
plt.show()

create_wav_from_audio(fenetre, sample_rate, "D:/Documents/uni/S5/APP3/note_basson_plus_sinus_1000_Hz_filtre_modifier.wav")