#! /usr/bin/python3


from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Duration of capture
duration_secs = 2

# sample rate
sample_rate_Hz = 50

# Return evenly spaced numbers over a specified interval.
# Start, Stop, Number of points
t = np.linspace(0.0, duration_secs, duration_secs*sample_rate_Hz)


# Recall trig function: A*sin(2pi*f*t + p); A = amplitude, f = frequency, p = phase
# Also "recall" that I = cos(phi), Q = sin(phi); 
def gen_IQ_signal(amplitude, freq, phase, time_array):
    I = amplitude*np.cos(2*np.pi*freq*t + phase)
    Q = amplitude*np.sin(2*np.pi*freq*t + phase)

    return I + 1j*Q

complex_signal = gen_IQ_signal(1,1,0,t)

complex_signal += gen_IQ_signal(1, 4, 0, t)

# Plot the raw complex signal
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot(np.real(complex_signal), np.imag(complex_signal), t, "blue")
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_zlabel("Time")

# Plot the_fft
ax = fig.add_subplot(212)
the_fft = fft(complex_signal)
the_fft = np.absolute(the_fft) # The values are complex, need the absolute val if we want constituent sine magnitude
the_fft = the_fft * 2 # We lose half the amplitude because of the nyquist freq mirroring (aliasing I suppose)
the_fft = the_fft / len(the_fft) # Just gotta do this because of the math of the FFT

bin_size = sample_rate_Hz / len(the_fft)

# Draw the nyquist frequency (we ignor everything past this freq)
ax.axvline(x=(sample_rate_Hz / 2))
ax.plot(np.array(range(0, len(the_fft)))*bin_size, the_fft)

plt.show()
