#! /usr/bin/python3
import h5py
import numpy as np
from scipy.fftpack import fft
import sys
import random
import math
import timeit # For time testing

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

# Use get_next_train_batch, or get_next_test_batch
# Will return (features, one hot encoded labels)
class Deepsig_Accessor:
    def __init__(self, modulations, SNRs, train_test_ratio, batch_size, throw_after_epoch=False, shuffle=True):
        self.batch_size = batch_size
        self.throw_after_epoch = throw_after_epoch

        self.current_test_batch_number = 0
        self.test_indices = []
        self.num_test_batches = None

        self.current_train_batch_number = 0
        self.train_indices = []
        self.num_train_batches = None

        random.seed(1337)

        self.deep_sig_file = h5py.File(deepsig_filename, 'r')
        deepsig_data_end_index = len(self.deep_sig_file["X"]) - 1

        if batch_size == None:
            self.batch_size = len(self.deep_sig_file["X"])

        # Build our indices, keep a list of all indices that match our targets
        target_indices = []
        for mod in modulations:
            for snr in SNRs:
                print("Fetching %s at %ddB" % (mod, snr))
                boundaries = find_boundaries_of_modulation(
                    self.deep_sig_file, class_map[mod], 0, deepsig_data_end_index, True)
                boundaries = find_boundaries_of_SNR(
                    self.deep_sig_file, snr, boundaries[0], boundaries[1], True)

                assert(len(boundaries) == 2)

                print(boundaries)
                
                # Build up our indices, +1 the end because our boundaries are inclusive while range is not
                target_indices += list(range(boundaries[0], boundaries[1]+1))
        
        # Shuffle them immediately
        if shuffle:
            random.shuffle(target_indices)
        else:
            print("[Deepsign_Accessor] - Warning, data will not be shuffled! (You probably don't want this)")

        # Split the data into training and test
        target_indices_split_point = int(len(target_indices)*train_test_ratio)

        self.train_indices = target_indices[0:target_indices_split_point]
        self.test_indices  = target_indices[target_indices_split_point:]

        # print("[Deepsig_Accessor] - Total num training samples: %d" % len(self.train_indices))
        # print("[Deepsig_Accessor] - Total num test samples: %d" % len(self.test_indices))

        self.num_train_batches = math.ceil(len(self.train_indices)/self.batch_size)
        self.num_test_batches = math.ceil(len(self.test_indices)/self.batch_size)

    # Helper function used by both train and test
    def compute_batch_indices(self, current_batch_number, indices_list, num_batches):
        assert(current_batch_number < num_batches)

        # Both begin and end are inclusive
        begin = current_batch_number*self.batch_size

        batch_indices_list = []
        if current_batch_number != num_batches-1:
            # Note, because list splising is exclusive on last item, we don't subtract by 1
            end = begin+self.batch_size
            batch_indices_list = indices_list[begin:end]
        else:  # Edge case for last batch
            batch_indices_list = indices_list[begin:]

        return batch_indices_list

    def generate_batch_output_from_indices_list(self, indices):
        MAX_CACHE_SIZE = 4095 #This is the size of one SNR group
        ret_iq = []
        ret_labels = []

        # print("[Deepsign_Accessor] - generate_batch_output_from_indices_list")
        # Begin the cache bullshit
        # We want to break the target indices into somewhat contiguous regions so that we can 
        # access efficiently form the shitty hdf5 file
        indices.sort()

        j = 0
        while j < len(indices):

            # Calculate the range of the erm, range, from the hdf5 file we will read
            cache_start_index = indices[j]
            cache_end_index   = cache_start_index
            indices_in_this_cache = [cache_start_index] # First index requested is of course in the line
            
            j += 1
            while (j < len(indices)) and (indices[j] - cache_start_index <= MAX_CACHE_SIZE):
                cache_end_index = indices[j]
                indices_in_this_cache.append(indices[j])
                j += 1
            
            # print("[Deepsign_Accessor] - new cache - range: [%d,%d], num indices: [%d]" 
            #     % (cache_start_index, cache_end_index, len(indices_in_this_cache)))

            # At this point we have our maximum cache size (or have hit the end of our indices)
            cache_x = self.deep_sig_file["X"][cache_start_index:cache_end_index+1]
            cache_y = self.deep_sig_file["Y"][cache_start_index:cache_end_index+1]

            for i in indices_in_this_cache:
                # Need to offset that since we're indexing into our cache instead of the file
                cache_index = i - cache_start_index 

                iq = cache_x[cache_index]
                flattened_iq = flatten_data_sample(iq)
                one_hot_encoding = cache_y[cache_index]

                ret_iq.append(flattened_iq)
                ret_labels.append(one_hot_encoding)
        
        return (ret_iq, ret_labels)

    def get_next_train_batch(self):
        if self.current_train_batch_number == self.num_train_batches:
            if self.throw_after_epoch:
                self.current_train_batch_number = 0
                raise StopIteration
            self.current_train_batch_number = 0
        # print("Getting train batch %d" % self.current_train_batch_number)
        # A list of indices of the deepsig file that correspond to the current batch
        batch_indices_list = self.compute_batch_indices(self.current_train_batch_number, self.train_indices, self.num_train_batches)

        # Increment our current batch number and wrap if necessary, indicating an epoch
        self.current_train_batch_number += 1

        return self.generate_batch_output_from_indices_list(batch_indices_list)
        
    def get_next_test_batch(self):
        if self.current_test_batch_number == self.num_test_batches:
            if self.throw_after_epoch:
                self.current_test_batch_number = 0
                raise StopIteration
            self.current_test = 0
        # print("Getting test batch %d" % self.current_test_batch_number)
        # A list of indices of the deepsig file that correspond to the current batch
        batch_indices_list = self.compute_batch_indices(
            self.current_test_batch_number, self.test_indices, self.num_test_batches)

        # Increment our current batch number and wrap if necessary, indicating an epoch
        self.current_test_batch_number += 1


        return self.generate_batch_output_from_indices_list(batch_indices_list)

    def get_total_num_training_samples(self):
        return len(self.train_indices)


    def get_total_num_testing_samples(self):
            return len(self.test_indices)
    
    def get_training_generator(self):
        for _ in range(0, self.num_train_batches):
            current_batch = self.get_next_train_batch()
            for i in range(0,len(current_batch[0])):
                yield (current_batch[0][i], current_batch[1][i])

    def get_testing_generator(self):
        for _ in range(0, self.num_test_batches):
            current_batch = self.get_next_test_batch()
            for i in range(0, len(current_batch[0])):
                yield (current_batch[0][i], current_batch[1][i])
        

if __name__ == '__main__':
    modulation_targets = '32QAM', 'FM'
    snr_targets = [30]

    ds_accessor = Deepsig_Accessor(
        modulation_targets, snr_targets, 0.75, batch_size=200, throw_after_epoch=True, shuffle=True)

    ds_training_generator = ds_accessor.get_training_generator()

    train_total_len = 0
    for t in ds_training_generator:
        train_total_len += 1
    
    print("Total train len: %d" % train_total_len)
    print("Should have training samples: %d" % ds_accessor.get_total_num_training_samples())
    #     print("Should have testing samples: %d" % ds_accessor.get_total_num_testing_samples())



    # def old_school_time_trial():
    #     dataset = get_data_samples(modulation_targets, snr_targets)
    #     train_x = []
    #     train_y = []
    #     test_x = []
    #     test_y = []
    #     set_split_point = int(len(dataset)*0.75)

    #     for i in range(0, set_split_point):
    #         train_x.append(dataset[i][0])
    #         train_y.append(dataset[i][2])

    #     for i in range(set_split_point, len(dataset)):
    #         test_x.append(dataset[i][0])
    #         test_y.append(dataset[i][2])

    #     print("Num old school training samples: %d" % len(train_x))
    #     print("Num old school test samples: %d" % len(test_x))

    # def ds_accessor_time_trial():

    #     ds_accessor = Deepsig_Accessor(
    #         modulation_targets, snr_targets, 0.75, batch_size=200, throw_after_epoch=True, shuffle=True)
        
    #     all_train_batches_lens = 0
    #     all_test_batches_lens  = 0

    #     print("Getting all training batches")
    #     while(True):
    #         try:
    #             all_train_batches_lens += len(ds_accessor.get_next_train_batch()[0])
    #         except StopIteration:
    #             break
        
    #     print("Getting all testing batches")
    #     while(True):
    #         try:
    #             all_test_batches_lens += len(ds_accessor.get_next_test_batch()[0])
    #         except StopIteration:
    #             break
        

    #     print("Num new school training samples: %d" % all_train_batches_lens)
    #     print("Num new school test samples: %d" % all_test_batches_lens)
    #     print("Should have training samples: %d" % ds_accessor.get_total_num_training_samples())
    #     print("Should have testing samples: %d" % ds_accessor.get_total_num_testing_samples())

    #     assert(all_train_batches_lens == ds_accessor.get_total_num_training_samples())
    #     assert(all_test_batches_lens == ds_accessor.get_total_num_testing_samples())
    #     print("DS accessor working as intended!")

    # print("Old School time: %f" % timeit.timeit(old_school_time_trial, number=1))
    # print("ds_accessor time: %f" % timeit.timeit(ds_accessor_time_trial, number=1000))

    # # Just leave this for debugging
    # f = h5py.File(deepsig_filename, 'r')
