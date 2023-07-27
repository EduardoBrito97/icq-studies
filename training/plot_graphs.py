import numpy as np
import matplotlib.pyplot as plt

def plot_graph(list_of_x, list_of_y, labelX, labelY):
    plt.plot(list_of_x, list_of_y, color="red", marker="o",  linestyle="")
    ax = plt.gca()
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='x', colors='red')
    
    ax.set_xlabel(labelX)
    ax.xaxis.label.set_color('red')

    ax.set_ylabel(labelY)
    ax.yaxis.label.set_color('red')
    
    plt.xticks(np.arange(min(list_of_x) * (-1), max(list_of_x), 50))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.show()


def plot_graph_mult_lines(list_of_xs, list_of_y, labelX, labelY):
    colors = ["red", "black", "blue", "yellow", "gray"]
    for index, list_of_x in enumerate(list_of_xs):
        plt.plot(list_of_x, list_of_y, color=colors[index], marker="o",  linestyle="-", label="Class " + str(index))
    ax = plt.gca()
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='x', colors='red')

    ax.set_xlabel(labelX)
    ax.xaxis.label.set_color('red')

    ax.set_ylabel(labelY)
    ax.yaxis.label.set_color('red')

    plt.legend(["Class " + str(i) for i in range(len(list_of_xs))])
    plt.yticks(np.arange(0, 1, 0.1))
    plt.show()