import csv, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

output_dir = "continouse_dataset/"
file_path = output_dir + 'loss_history.csv'
fig, (ax1, ax2) = plt.subplots(2)

def readCSV(path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
    return x_data, y_data


#plt.savefig(output_dir + 'loss_graph.png')

def animate(i):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        xar = []
        yar = []
        for row in reader:
            xar.append(float(row[0]))
            yar.append(float(row[1]))
        ax1.clear()
        ax2.clear()
        ax1.plot(xar,yar)
        ax2.plot(xar,yar)
        ax2.autoscale()
        ax1.autoscale()
        ax2.set_xlim(len(xar) - 200, len(xar) + 10)
        ax1.set_ylim(0.0, ax1.get_ylim()[1])


if __name__ == "__main__":

    ani = animation.FuncAnimation(fig, animate, interval=1, cache_frame_data=False)
    plt.show()
