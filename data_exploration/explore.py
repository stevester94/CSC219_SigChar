#! /usr/bin/python3
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import sys

filename = "deepsig_data/GOLD_XYZ_OSC.0001_1024.hdf5"

class_map = {
 '32PSK':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '16APSK':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '32QAM':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'FM':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'GMSK':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '32APSK':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'OQPSK':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '8ASK':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'BPSK':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '8PSK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'AM-SSB-SC':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '4ASK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '16PSK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '64APSK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '128QAM':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 '128APSK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 'AM-DSB-SC':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 'AM-SSB-WC':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 '64QAM':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 'QPSK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 '256QAM':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 'AM-DSB-WC':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 'OOK':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 '16QAM':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

def find_boundaries_of_modulation(hdf5_file, modulation, start_index, end_index):
    target_class = modulation.index(1)

    middle_index = int((end_index + start_index)/2)


    class_of_start = hdf5_file['Y'][start_index].tolist().index(1)
    class_of_end = hdf5_file['Y'][end_index].tolist().index(1)

    if class_of_start > target_class or class_of_end < target_class:
        return None

    # Base case, we've narrowed it down
    if end_index - start_index == 1:
        if class_of_start == target_class: return start_index
        elif class_of_end   == target_class: return end_index
        else:
            print("Impossible base case")
            sys.exit(1)

    # Region where end boundary exists
    if class_of_start == target_class and class_of_end != target_class:
        end = find_boundaries_of_modulation(hdf5_file, modulation, start_index, middle_index)
        if end == None:
            end = find_boundaries_of_modulation(hdf5_file, modulation, middle_index, end_index)

        if end == None:
            print("Impossible case")
            sys.exit(1)

        # Special case where we were given the whole dataset and the modulation begins at the start
        if start_index == 0 and end_index == len(f["Y"])-1:
            return (0,end)
        else:
            return end

    # Region where begin boundary exists
    if class_of_start != target_class and class_of_end == target_class:
        begin = find_boundaries_of_modulation(hdf5_file, modulation, start_index, middle_index)
        if begin == None:
            begin = find_boundaries_of_modulation(hdf5_file, modulation, middle_index, end_index)

        if begin == None:
            print("Impossible case")
            sys.exit(1)

        # Special case where we were given the entire list and data occurs at the end (IE no end boundary)
        if start_index == 0 and end_index == len(f["Y"])-1:
            return (begin, len(f["Y"])-1)
        else:
            return begin
    
    # Region where both boundaries exist
    if class_of_start < target_class and class_of_end > target_class:
        begin = find_boundaries_of_modulation(hdf5_file, modulation, start_index, middle_index)

        # The first half contains nothing, meaning the second half must contain both
        if begin == None:
            return find_boundaries_of_modulation(hdf5_file, modulation, middle_index, end_index)

        # The first half contained both
        if isinstance(begin, tuple):
            return begin 

        # The fist half contained the begining, the second half must contain the end
        else:
            return (begin, find_boundaries_of_modulation(hdf5_file, modulation, middle_index, end_index))


         





def find_an_index_of_modulation(hdf5_file, modulation, start_index, end_index):
    target_class = modulation.index(1)
    index_of_middle_of_current = int((end_index + start_index)/2)
    class_of_middle_of_current = hdf5_file['Y'][index_of_middle_of_current].tolist().index(1)

    print("%d, %d, %d" % (start_index, index_of_middle_of_current, end_index))

    if class_of_middle_of_current == target_class:
        return index_of_middle_of_current
    elif class_of_middle_of_current > target_class:
        return find_an_index_of_modulation(hdf5_file, modulation, start_index, index_of_middle_of_current)
    elif class_of_middle_of_current < target_class:
        return find_an_index_of_modulation(hdf5_file, modulation, index_of_middle_of_current, end_index)
    else:
        print("IMPOSSIBLE CASE")

def get_data_sample(modulation):
    f = h5py.File(filename, 'r')
    ret_list = []
    # for index,mod_type in enumerate(f['Y']):
    #     if list(mod_type) == class_map[modulation]:
    #         # ret_list.append(d)
    #         pass
    #     print("percent complete %f" % (index / len(f['Y'])))
    
    # return ret_list

    index = find_an_index_of_modulation(f, class_map[modulation], 0, len(f['X']))
    
    return f['X'][index]

def flatten_data_sample(data_sample):
    return [d[0] + 1j*d[1] for d in data_sample]

f = h5py.File(filename, 'r')

# 32PSK, 16QAM
print(find_boundaries_of_modulation(f, class_map["16QAM"], 0, len(f['X'])-1))


sys.exit(0)
data = f['X'][0]
sample = get_data_sample("16QAM")
sample = flatten_data_sample(sample)

# Plot the raw complex signal
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot(np.real(sample), np.imag(sample), range(0,len(sample)), "blue")
ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.set_zlabel("Index")

# Plot the_fft
ax = fig.add_subplot(212)
print(sample)

the_fft = fft(sample)
the_fft = np.absolute(the_fft) # The values are complex, need the absolute val if we want constituent sine magnitude
the_fft = the_fft * 2 # We lose half the amplitude because of the nyquist freq mirroring (aliasing I suppose)
the_fft = the_fft / len(the_fft) # Just gotta do this because of the math of the FFT

# Draw the nyquist frequency (we ignor everything past this freq)
# ax.axvline(x=(sample_rate_Hz / 2))

print(the_fft)

ax.plot(the_fft)

plt.show()
