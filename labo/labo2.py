import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


fc = 2000
fe = 16000
N = [16, 32, 64]
B = 4096

m = [fc * n / fe for n in N]
k = [2 * n / fe for n in m]

H = [[], [], []]
H[0] = [1 if n <= m[0] or N[0] - n - 1 <= m[0] else 0 for n in range(N[0])]
H[1] = [1 if n <= m[1] or N[1] - n - 1 <= m[1] else 0 for n in range(N[1])]
H[2] = [1 if n <= m[2] or N[2] - n - 1 <= m[2] else 0 for n in range(N[2])]

h = [[], [], []]
h[0] = np.fft.ifft(H[0])
h[1] = np.fft.ifft(H[1])
h[2] = np.fft.ifft(H[2])

hz = [[], [], []]
hz[0] = [h[0][i] if i < len(h[0]) else 0 for i in range(B)]
hz[1] = [h[1][i] if i < len(h[1]) else 0 for i in range(B)]
hz[2] = [h[2][i] if i < len(h[2]) else 0 for i in range(B)]

Hz = [[], [], []]
Hz[0] = np.fft.fft(hz[0])
Hz[1] = np.fft.fft(hz[1])
Hz[2] = np.fft.fft(hz[2])

fig, axs = plt.subplots(3, 3)
axs[0][0].stem(range(N[0]), H[0])
axs[1][0].stem(range(N[0]), h[0].real)
axs[2][0].stem(range(B),    Hz[0].real)

axs[0][1].stem(range(N[1]), H[1])
axs[1][1].stem(range(N[1]), h[1].real)
axs[2][1].stem(range(B),    Hz[1].real)

axs[0][2].stem(range(N[2]), H[2])
axs[1][2].stem(range(N[2]), h[2].real)
axs[2][2].stem(range(B),    Hz[2].real)
plt.show()
