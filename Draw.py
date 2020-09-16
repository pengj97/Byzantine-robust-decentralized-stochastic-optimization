import numpy as np
import matplotlib.pyplot as plt
import pickle
import Config

set_iteration = []
for k in range(1, Config.optConfig['iterations']+1):
    if k % 200 == 0 or k == 1:
        set_iteration.append(k)
colors = ['orange', 'blue', 'green', 'red']
markers = ['s', '^', 'v', 'o']


def draw(attack):
    """
    Draw the curve of experimental results

    :param attack:
                   'wa': without Byzantine attacks,
                   'sv': same-value attacks
                   'sf': sign-flipping attacks
                   'noniid': sample-duplicating attacks in non-i.i.d. case
    """

    methods = ['dpsgd', 'bridge', 'byrdie', 'ours']
    labels = ['DPSGD', 'BRIDGE-S', 'ByRDiE-S', 'Our proposed']

    acc_list = []
    var_list = []

    for i in range(len(methods)):
        with open("paper-results/" + methods[i] + "-" + attack + ".pkl", "rb") as f:
            acc, var =pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i in range(len(methods)):
        plt.plot(set_iteration, acc_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)

    plt.figure(2)
    for i in range(len(methods)):
        plt.plot(set_iteration, var_list[i], color=colors[i], marker=markers[i], label=labels[i])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)

    plt.show()


def draw_imopp():
    """
    Plot the curve of experiment results of 'Impact of Penalty Parameter'
    :return:
    """
    values = [0, 0.001, 0.01, 0.12]
    acc_list = []
    var_list = []

    for i in range(len(values)):
        with open("results/imopp-"+ str(i+1) + ".pkl", "rb") as f:
            acc, var = pickle.load(f)
            acc_list.append(acc)
            var_list.append(var)

    plt.figure(1)
    for i in range(len(values)):
        plt.plot(set_iteration, acc_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Classification Accuracy', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.legend(fontsize=15)

    plt.figure(2)
    for i in range(len(values)):
        plt.plot(set_iteration, var_list[i], color=colors[i], marker=markers[i], label=r"$\lambda = $"+str(values[i]))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Variance', fontsize=15)
    plt.xlabel('Number of iterations', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=15)

    plt.show()


if __name__ == '__main__':
    # draw('wa')
    # draw('sv')
    # draw('sf')
    draw('noniid')
    # draw_imopp()


