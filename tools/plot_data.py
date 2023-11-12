import csv, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

output_dir = sys.argv[1]
file_path = output_dir + 'loss_history.csv'
fig, (ax1, ax2) = plt.subplots(2)

def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    return moving_averages

def readCSV(path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            x_data.append(float(count))
            y_data.append(float(row[1]))
            count = count + 1
    return x_data, y_data

def animate(i):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        xar = []
        yar = []
        count = 0
        for row in reader:
            xar.append(float(count))
            yar.append(float(row[1]))
            count = count + 1

    #with open(output_dir + 'checkpoint_history.csv', 'r') as csvfile:
    #    reader = csv.reader(csvfile)
    #    xckp = []
    #    yckp = []
    #    for row in reader:
    #        xckp.append(float(row[0]))
    #        yckp.append(float(row[1]))

    ax1.clear()
    ax2.clear()
    #ax1.plot(xckp, yckp)
    ax1.plot(xar,yar, linewidth=0.5)
    ax2.plot(xar,yar, linewidth=0.5)
    ax2.autoscale()
    ax1.autoscale()
    ax2.set_xlim(len(xar) - 200, len(xar) + 10)
    ax1.set_ylim(0.0, ax1.get_ylim()[1])

#    for x in xckp:
#        ax1.axvline(x=x, ymin=0, ymax=ax1.get_ylim()[1], ls=':', c=[1., 0., 0.], linewidth=0.8)
    ax1.grid(linestyle='-.', linewidth=0.5)
    ax2.grid(linestyle='-.', linewidth=0.5)


if __name__ == "__main__":
    ani = animation.FuncAnimation(fig, animate, interval=10, cache_frame_data=False)
    plt.show()
