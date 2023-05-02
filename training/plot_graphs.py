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
    
    plt.xticks(np.arange(min(list_of_x) * (-1), max(list_of_x), 1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.show()