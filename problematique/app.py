#filtre passebas
omega = np.pi /1000
#f = (𝜔̅ /2 )Fe 
fe = sample_rate
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
#avec sqrt(2)/2 on obtient 886 pour N et avec -3 db on on obtient 884 pour N nous allons prendre 884 pour N
x = np.arange(0, np.pi, np.pi/5000)
Hn = []	
for w in x:
    sumExp =0
    for n in range(0,N,1):
        sumExp += np.exp(-1j * w * n)
    val =  sumExp/N
    Hn.append(val)
hn = np.fft.ifft(Hn)
plt.plot(x, 20*np.log10(np.abs(Hn)))
plt.show()
plt.plot(x, hn)
plt.show()

#coupe bande
#filtre coupe-bande
N2=6000
fc2 = 2000
fcmin = 960
fcmax = 1040
fe2 = sample_rate_2
#like labo
mMin = (fcmin * N2) / fe2
mMax = (fcmax * N2) / fe2
# Creation de l'impulsion frequentielle du filtre
Hn = [1 if n < mMin or n > mMax else 0 for n in range(0,N2,1)]

# Reponse impulsionnelle du filtre
hn = np.fft.ifft(Hn)
x = np.arange(0, np.pi, np.pi/6000)
#sin de 1000hz
sin = np.sin(1000 * 2 * np.pi)
result = np.convolve(sin, hn)
plt.plot(range(N2), Hn)
plt.xlim(100,200)
plt.show()
plt.stem(range(N2), hn)
plt.xlim(0,200)
plt.show()
plt.plot( result)
plt.show()
#get data bassson
with wave.open('D:/Documents/uni/S5/APP3/note_basson_plus_sinus_1000_Hz_filtre_modifier.wav') as wav:   # Ouverture du fichier audio
    sample_rate_2 = wav.getframerate()
    frames_2      = wav.readframes(-1)
    frames_2      = np.frombuffer(frames_2, dtype=np.int16)
    # Normalize at 1
    # max_amp2 = np.amax(frames_2)
    # frames2 = np.divide(frames_2, max_amp2)
    plt.plot(frames_2)
    plt.show()
#filtre coupe-bande
N2=6000
fe2 = sample_rate_2
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
plt.plot(data,hLp)
plt.show()

hnBasson =[dn[i] - np.multiply(2 * hLp[i] , np.cos(w0 * data[i])) for i in range(0,N2,1)]

plt.plot(data,hnBasson)
plt.xlim(-100,100)
plt.show()

HnBasson = np.fft.fft(hnBasson)
cb_freqs = np.fft.fftfreq(len(hnBasson),d=1/fe2)
plt.plot(cb_freqs[:500],np.abs(HnBasson[:500]))
plt.show()

# #like labo
# N2=6000
# fe2 = sample_rate_2
# #créer les mMin et mMax
# fcmin = 960
# fcmax = 1040


# mMin = (fcmin * N2) / fe2
# mMax = (fcmax * N2) / fe2
# # Creation de l'impulsion frequentielle du filtre
# HnBasson = [1 if n < mMin or n > mMax else 0 for n in range(0,N2,1)]

# # Reponse impulsionnelle du filtre
# hnBasson = np.fft.ifft(HnBasson)
# x = np.arange(0, np.pi, np.pi/6000)
# sin = np.sin(1000 * 2 * np.pi)
# result = np.convolve(sin, hnBasson)


# plt.plot(range(N2), HnBasson)
# plt.xlim(100,200)
# plt.show()
# plt.stem(range(N2), hnBasson)
# plt.xlim(0,200)
# plt.show()
# plt.plot(result)
# plt.show()

#convo + fenetre et creer son
#Convolution du signal basson avec le filtre
result = np.convolve(frames_2, hnBasson)
fenetre = np.hanning(len(result))*result
plt.plot(fenetre)
plt.show()
def create_wav_from_audio(audio, sampleRate, filename):
    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes   = len(audio)
        wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

        for sample in audio:
            wav.writeframes(struct.pack('h', np.int16(sample)))

create_wav_from_audio(fenetre, sample_rate_2, "D:/Documents/uni/S5/APP3/note_basson_plus_sinus_1000_Hz_filtre_modifier.wav")