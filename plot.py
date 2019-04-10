import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)


def load_log(file):

    results = []
    with open(file) as f:
        for line in f:
            if '*Prec@1' in line:
                line = line.split()
                results.append(float(line[-1]))
    return results


shows = dict()
shows['vgg16'] = load_log('log_vgg16')
shows['googlenet'] = load_log('log_googlenet')
shows['alexnet'] = load_log('log_alexnet')
shows['resnet152'] = load_log('log_resnet152')

for key in sorted(shows.keys()):
    epochs = np.arange(1, 1+len(shows[key]))
    plt.plot(epochs, shows[key], label='{key}:{np.max(shows[key]}')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
