#! /usr/bin/python3
import h5py
import numpy as np
from scipy.fftpack import fft
import sys

deepsig_filename = "../../data_exploration/deepsig_data/GOLD_XYZ_OSC.0001_1024.hdf5"

valid_snr = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2)

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
 '16QAM':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

def find_boundaries_of_modulation(hdf5_file, modulation, start_index, end_index, first_call=False):
    target_class = modulation.index(1)

    middle_index = int((end_index + start_index)/2)


    class_of_start = hdf5_file['Y'][start_index].tolist().index(1)
    class_of_end = hdf5_file['Y'][end_index].tolist().index(1)

    if class_of_start > target_class or class_of_end < target_class:
        return None

    # Base case, we've narrowed it down
    if end_index - start_index == 1:
        if class_of_start == class_of_end:
            print("Hit the weird case of start and end being equal")
            return start_index

        elif class_of_start == target_class: return start_index
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
        if first_call:
            return (start_index,end)
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
        if first_call:
            return (begin, end_index)
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


         
def find_boundaries_of_SNR(hdf5_file, SNR, start_index, end_index, first_call=False):
    if SNR not in valid_snr:
        print("Invalid SNR requested %d" % SNR)
        return
    target_class = SNR

    middle_index = int((end_index + start_index)/2)


    class_of_start = hdf5_file['Z'][start_index]
    class_of_end = hdf5_file['Z'][end_index]

    if class_of_start > target_class or class_of_end < target_class:
        return None

    # Base case, we've narrowed it down
    if end_index - start_index == 1:
        if class_of_start == class_of_end:
            print("Hit the weird case of start and end being equal")
            return start_index
        elif class_of_start == target_class: return start_index
        elif class_of_end  == target_class: return end_index
        else:
            print("Impossible base case")
            sys.exit(1)

    # Region where end boundary exists
    if class_of_start == target_class and class_of_end != target_class:
        end = find_boundaries_of_SNR(hdf5_file, SNR, start_index, middle_index)
        if end == None:
            end = find_boundaries_of_SNR(hdf5_file, SNR, middle_index, end_index)

        if end == None:
            print("Impossible case")
            sys.exit(1)

        # Special case where we were given the whole dataset and the modulation begins at the start
        if first_call:
            return (start_index,end)
        else:
            return end

    # Region where begin boundary exists
    if class_of_start != target_class and class_of_end == target_class:
        begin = find_boundaries_of_SNR(hdf5_file, SNR, start_index, middle_index)
        if begin == None:
            begin = find_boundaries_of_SNR(hdf5_file, SNR, middle_index, end_index)

        if begin == None:
            print("Impossible case")
            sys.exit(1)

        # Special case where we were given the entire list and data occurs at the end (IE no end boundary)
        if first_call:
            return (begin, end_index)
        else:
            return begin
    
    # Region where both boundaries exist
    if class_of_start < target_class and class_of_end > target_class:
        begin = find_boundaries_of_SNR(hdf5_file, SNR, start_index, middle_index)

        # The first half contains nothing, meaning the second half must contain both
        if begin == None:
            return find_boundaries_of_SNR(hdf5_file, SNR, middle_index, end_index)

        # The first half contained both
        if isinstance(begin, tuple):
            return begin 

        # The fist half contained the begining, the second half must contain the end
        else:
            return (begin, find_boundaries_of_SNR(hdf5_file, SNR, middle_index, end_index))






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


def flatten_data_sample(data_sample, flatten_to_IQ=False):
    # This actually turns it into IQ samples
    if flatten_to_IQ:
        return [d[0] + 1j*d[1] for d in data_sample] # The multiply by j is slow as fuck

    else:
        # This just does a straight flatten, it's also fucking fast
        return data_sample.flatten()


# Accepts a list of modulations and SNRs
# Returns a list of tuples, each of the type (numpy_array(the IQ), modulation (string representation), modulation (one hot), SNR)
def get_data_samples(modulations, SNRs, flatten_to_IQ=False):
    f = h5py.File(deepsig_filename, 'r')

    deepsig_data_end_index = len(f["X"]) - 1

    ret_list = []

    for mod in modulations:
        for snr in SNRs:
            print("Fetching %s at %ddB" % (mod, snr))
            boundaries = find_boundaries_of_modulation(
                f, class_map[mod], 0, deepsig_data_end_index, True)
            boundaries = find_boundaries_of_SNR(f, snr, boundaries[0], boundaries[1], True)

            assert(len(boundaries) == 2)

            # Get the data, and flatten it because the internal representation doesn't do complex
            deepsig_IQ = f["X"][boundaries[0]:boundaries[1]]

            print("Flattening data")
            flattened_IQ = []
            for IQ in deepsig_IQ:
                flattened_IQ.append(flatten_data_sample(IQ, flatten_to_IQ))
            
            deepsig_modulation_one_hot = f["Y"][boundaries[0]:boundaries[1]]

            # Verify everything is the same length (They should be parallel)
            assert(len(flattened_IQ) == len(deepsig_modulation_one_hot) == len(deepsig_IQ))

            # Build the list
            for i in range(0, len(flattened_IQ)):
                list_item = (flattened_IQ[i], mod,
                             deepsig_modulation_one_hot[i], snr)
                ret_list.append(list_item)

    return ret_list


if __name__ == '__main__':
    # modulation_boundaries = find_boundaries_of_modulation(f, class_map["32QAM"], 0, len(f['X'])-1, True)
    # SNR_boundaries = find_boundaries_of_SNR(f, 30, modulation_boundaries[0], modulation_boundaries[1], True)

    # data_samples = get_data_samples(["32PSK", "OOK", "16QAM"], [30])
    data_samples = get_data_samples(["32PSK"], [30], flatten_to_IQ=False)
    # data_samples = get_data_samples([], [])
    print("Length of data: %d" % len(data_samples))

    f = h5py.File(deepsig_filename, 'r')

