#############################
# Target dataset parameters
#############################
all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

offenders = ["64QAM", "AM-SSB-WC"]

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
limited_snr = [-20, -10, 0, 10, 20, 30]
thirty_snr = [30]
low_snr = [-20, -10]
medium_snr = [0, 10]
high_snr = [20, 30]

OTA_ARCH = {
    "learning_rate": 0.001,
    "num_train_epochs": 50,
    "batch_size": 50, # Orig 100
    "target": (all_modulation_targets, limited_snr),
    "label": "Real OTA arch",
    "network_conv_settings": [
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
        {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    ],
    "network_fc_settings": [
        {"fc_num_nodes": 128},
        {"fc_num_nodes": 128}
    ]
}


def get_test_cases():
    test_cases = []

    # test_cases.append(OTA_ARCH)


    # case = {
    #     "label": "OTA 256 filters",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 512 filters",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 512, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 128 filters of size 2",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 128, 4 conv",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256 filters, 4+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 128 filters, 4+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 128 filters, 8+1 hidden of 64",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64},
    #         {"fc_num_nodes": 64}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "Deep perceptron, 8+1 hidden of 256",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 10E",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 10,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 20E",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 20,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2 50E",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 75E",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 75,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "Toy 1 epoch GPU",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 1,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, thirty_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 10E, LOW",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 10,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, low_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 10E, MEDIUM",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 10,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, medium_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256, filters 5,2, 10E, HIGH",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 10,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, high_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    case = {
        "label": "SEPARATOR OTA 256, filters 5,2, 10E, ALL",
        "learning_rate": 0.001,
        "num_train_epochs": 10,
        "batch_size": 100,
        "target": (offenders, all_snr_targets),
        "network_conv_settings": [
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128}
        ]
    }
    test_cases.append(case)

    return test_cases

def get_toy_test_cases():
    test_cases = []

    case = {
        "label": "Toy 1 epoch CPU",
        "learning_rate": 0.001,
        "num_train_epochs": 1,
        "batch_size": 100,
        "target": (all_modulation_targets, thirty_snr),
        "network_conv_settings": [
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 5, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 2, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128}
        ]
    }
    test_cases.append(case)

    # parameters_dict = {
    #     "label": "toy case2",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 1,
    #     "batch_size": 100,
    #     "target": (all_modulation_targets, thirty_snr),
    #     "network_conv_settings": [
    #     ],
    #     "network_fc_settings": [
    #     ]
    # }
    # test_cases.append(parameters_dict)

    return test_cases

def get_robust_easy():
    test_cases = []


    parameters_dict = {
        "learning_rate": 0.001,
        "num_train_epochs": 10,
        "batch_size": 100,
        "target": (subset_modulation_targets, thirty_snr),
        "label": "robust toy",
        "network_conv_settings": [
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128}
        ]
    }
    test_cases.append(parameters_dict)

    return test_cases


def get_offenders():
    test_cases = []

    # case = {
    #     "label": "SEPARATOR OTA 128 filters, 4+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #         "target": (offenders, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "SEPARATOR OTA 128 filters, 2+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (offenders, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 128, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "SEPARATOR OTA 256 filters, 2+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (offenders, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "SEPARATOR OTA 256 filters, 4+1 hidden of 128",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (offenders, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128},
    #         {"fc_num_nodes": 128}
    #     ]
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "SEPARATOR OTA 256 filters, 4+1 hidden of 256",
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 50,
    #     "batch_size": 100,
    #     "target": (offenders, limited_snr),
    #     "network_conv_settings": [
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
    #         {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
    #     ],
    #     "network_fc_settings": [
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256},
    #         {"fc_num_nodes": 256}
    #     ]
    # }
    # test_cases.append(case)

    case = {
        "label": "SEPARATOR OTA 256 filters, 4+1 hidden of 256, 100 epochs",
        "learning_rate": 0.001,
        "num_train_epochs": 100,
        "batch_size": 100,
        "target": (offenders, limited_snr),
        "network_conv_settings": [
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 256, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 256},
            {"fc_num_nodes": 256},
            {"fc_num_nodes": 256},
            {"fc_num_nodes": 256}
        ]
    }
    test_cases.append(case)

    return test_cases



