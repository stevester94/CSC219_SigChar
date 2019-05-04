import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


d = np.array(
    [[402,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
     [0, 387,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
     [0,  21, 412,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
     [0,   0,   0, 370,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  14],
     [0,   0,   0,   0,  75,   0, 300,  52,   4,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2],
     [0,   0,   0,   0,  34,   0, 321,  55,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   2],
     [0,   0,   0,   0,  42,   0, 301,  37,   2,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2],
     [0,   0,   0,   0,  37,   0, 323,  57,   1,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1],
     [0,   0,   0,   0,  21,   0,   0,   1, 375,   0,   0,   0,   0,  11,   0,   0,   0,   0,   0,   0,   0,   0,   0,  20],
     [0,   0,   0,   0,   0,   0,   0,   0,   0, 331,   8,   3,   0,  5,   0,   2,  16,   0,   0,   0,   0,   0,   0,  17],
     [0,   0,   0,   0,   0,   0,   0,   0,  13,  87,  14,   1,   0,  80,   0,  68, 125,   0,   0,   0,   0,   0,   0,  14],
     [0,   0,   0,   0,   0,   0,   0,   0,   1, 177,   8,   5,   0,  13,   0,  33, 168,   0,   0,   0,   0,   0,   0,   6],
     [0,   0,   0,   0,   0,   0,   0,   0,  49,  22,   4,   1,   0,  124,   0,  91, 100,   0,   0,   0,   0,   0,   0,   5],
     [0,   0,   0,   0,   0,   0,   0,   0,  27,  67,  13,   0,   0,  118,   0,  80,  99,   0,   0,   0,   0,   0,   0,  23],
     [0,   0,   0,   0,   0,   0,   0,   0,   3,  69,   2,   1,   0,  37,   0,  62, 225,   0,   0,   0,   0,   0,   0,   3],
     [0,   0,   0,   0,   0,   0,   0,   0,  14,  63,  13,   2,   0,  62,   0,  97, 176,   0,   0,   0,   0,   0,   0,   9],
     [0,   0,   0,   0,   0,   0,   0,   0,   2,  53,   5,   3,   0,  23,   0,  47, 257,   0,   0,   0,   0,   0,   0,   1],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0, 209, 174,   0,   0,   0,   0,   0],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,  91, 307,   0,   0,   0,   0,   0],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0, 384,  26,   0,   0,   0],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,  84, 355,   0,   0,   0],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0, 426,   0,   0],
     [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   1, 415,   0],
     [0,   0,   0,   0,   5,   0,   0,   0,   8,   0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 405]],
    dtype=np.int32)

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']

def plot_confusion_matrix(confusion, labels):
   # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)



    # cax = ax.matshow(d, cmap='bone')
    cax = ax.matshow(confusion, cmap='Purples')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + labels, rotation=90)
    ax.set_yticklabels([""] + labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


    # This is very hack-ish
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor')


    # Set labels
    ax.set_xlabel("Predicted Modulation")
    ax.set_ylabel("Actual Modulation")



    plt.show()

plot_confusion_matrix(d, all_modulation_targets)

