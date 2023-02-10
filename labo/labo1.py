import matplotlib.pyplot as plt
import numpy as np

N = 100

x1 = np.sin([0.1 * n * np.pi + np.pi/4 for n in range(N)])
x2 = [1 if mod == 0 else -1 for mod in [n % 2 for n in range(N)]]
x3 = [1 if n == 10 else 0 for n in range(N)]

X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

fig, axs = plt.subplots(3)
axs[0].stem(range(N), x1)
axs[1].stem(range(N), np.sqrt(X1.real**2 + X1.imag**2))
axs[2].stem(range(N), np.arctan(X1.imag / X1.real))
plt.show()

fig, axs = plt.subplots(3)
axs[0].stem(range(N), x2)
axs[1].stem(range(N), np.sqrt(X2.real**2 + X2.imag**2))
axs[2].stem(range(N), np.arctan(X2.imag / X2.real))
plt.show()

fig, axs = plt.subplots(3)
axs[0].stem(range(N), x3)
axs[1].stem(range(N), np.sqrt(X3.real**2 + X3.imag**2))
axs[2].stem(range(N), np.arctan(X3.imag / X3.real))
plt.show()
