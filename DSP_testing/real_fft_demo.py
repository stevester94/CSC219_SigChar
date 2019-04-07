#! /usr/bin/python3


from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Duration of capture
duration_secs = 5

# sample rate
sample_rate_Hz = 10

# Return evenly spaced numbers over a specified interval.
# Start, Stop, Number of points
t = np.linspace(0.0, duration_secs, duration_secs*sample_rate_Hz)


# Recall trig function: A*sin(2pi*f*t + p); A = amplitude, f = frequency, p = phase
amplitude_1 = 2
freq_1 = 1
phase_1 = 0
y1 = amplitude_1*np.sin(2*np.pi*freq_1*t + phase_1)

amplitude_2 = 1
freq_2 = 4
phase_2 = 0
y2 = amplitude_2*np.sin(2*np.pi*freq_2*t + phase_2)

y = y1+y2

plt.figure()
plt.subplot(211)


the_fft = fft(y)
the_fft = np.absolute(the_fft) # The values are complex, need the absolute val if we want constituent sine magnitude
the_fft = the_fft * 2 # We lose half the amplitude because of the nyquist freq mirroring (aliasing I suppose)
the_fft = the_fft / len(the_fft) # Just gotta do this because of the math of the FFT

bin_size = sample_rate_Hz / len(the_fft)

# Draw the nyquist frequency (we ignor everything past this freq)
plt.axvline(x=(sample_rate_Hz / 2))
plt.plot(np.array(range(0, len(the_fft)))*bin_size, the_fft)



print("Len the_fft %d" % len(the_fft))
print(the_fft)

plt.subplot(212)
plt.plot(t, y)

plt.show()
