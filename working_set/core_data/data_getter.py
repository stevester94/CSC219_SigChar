import deepsig_accessor as ds_accessor
import struct

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]


all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

# Will fetch only one sample for each entry
targets = [
    ('32PSK', 30),
    ('16APSK', 30),
    ('32QAM', 30),
    ('FM', 30),
    ('GMSK', 30),
    ('32APSK', 30),
    ('OQPSK', 30),
    ('8ASK', 30),
    ('BPSK', 30),
    ('8PSK', 30),
    ('AM-SSB-SC', 30),
    ('4ASK', 30),
    ('16PSK', 30),
    ('64APSK', 30),
    ('128QAM', 30),
    ('128APSK', 30),
    ('AM-DSB-SC', 30),
    ('AM-SSB-WC', 30),
    ('64QAM', 30),
    ('QPSK', 30),
    ('256QAM', 30),
    ('AM-DSB-WC', 30),
    ('OOK', 30),
    ('16QAM', 30)
]


for target in targets:
    # Have to make them into lists since that's what the ds_accesor requires
    dataset = ds_accessor.get_data_samples([target[0]], [target[1]])

    IQ = dataset[0][0]
    print(IQ)
    blob = struct.pack("2048f", *IQ)
    with open(target[0] + "_" + str(target[1]) + ".bin", "wb") as file_out:
        file_out.write(blob)
