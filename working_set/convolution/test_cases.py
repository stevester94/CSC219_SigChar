#############################
# Target dataset parameters
#############################
all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
limited_snr = [-20, -10, 0, 10, 20, 30]
high_snr = [24, 26, 28, 30]
thirty_snr = [30]



def get_test_cases():
    test_cases = []

    # parameters_dict = {
    #     "learning_rate": 0.001,
    #     "num_train_epochs": 100,
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
    # test_cases.append(parameters_dict)

    # This one is broken
    parameters_dict = {
        "label": "Lots of smaller filters, more FC",
        "learning_rate": 0.001,
        "num_train_epochs": 100,
        "batch_size": 100,
        "target": (all_modulation_targets, limited_snr),
        "network_conv_settings": [
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128}
        ]
    }
    test_cases.append(parameters_dict)

    parameters_dict = {
        "label": "Few smaller filters",
        "learning_rate": 0.001,
        "num_train_epochs": 100,
        "batch_size": 100,
        "target": (all_modulation_targets, limited_snr),
        "network_conv_settings": [
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2},
            {"conv_num_filters": 32, "conv_kernel_size": 3, "max_pool_stride": 2, "max_pool_kernel_size": 2}
        ],
        "network_fc_settings": [
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128},
            {"fc_num_nodes": 128}
        ]
    }
    test_cases.append(parameters_dict)

    parameters_dict = {
        "label": "More nodes in FC",
        "learning_rate": 0.001,
        "num_train_epochs": 100,
        "batch_size": 100,
        "target": (all_modulation_targets, limited_snr),
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
            {"fc_num_nodes": 256},
            {"fc_num_nodes": 256}
        ]
    }
    test_cases.append(parameters_dict)

    parameters_dict = {
        "label": "Even more nodes in FC",
        "learning_rate": 0.001,
        "num_train_epochs": 100,
        "batch_size": 100,
        "target": (all_modulation_targets, limited_snr),
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
            {"fc_num_nodes": 512},
            {"fc_num_nodes": 512}
        ]
    }
    test_cases.append(parameters_dict)

    return test_cases

def get_toy_test_cases():
    test_cases = []

    parameters_dict = {
        "label": "toy case1",
        "learning_rate": 0.001,
        "num_train_epochs": 1,
        "batch_size": 100,
        "target": (all_modulation_targets, thirty_snr),
        "network_conv_settings": [
        ],
        "network_fc_settings": [
        ]
    }
    test_cases.append(parameters_dict)

    parameters_dict = {
        "label": "toy case2",
        "learning_rate": 0.001,
        "num_train_epochs": 1,
        "batch_size": 100,
        "target": (all_modulation_targets, thirty_snr),
        "network_conv_settings": [
        ],
        "network_fc_settings": [
        ]
    }
    test_cases.append(parameters_dict)

    return test_cases
