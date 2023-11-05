import csv, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

output_dir = sys.argv[1]
file_path = output_dir + 'labels.csv'
fig, (ax1) = plt.subplots(1)

def readCSV(path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
    return x_data, y_data

if __name__ == "__main__":
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        xar = []
        yar = []
        for row in reader:
            xar.append(float(row[1]))
            yar.append(float(row[2]))
    ax1.scatter(xar,yar, color=[[1., 0., 0.]])
    ax1.autoscale()
    ax1.grid(linestyle='-.', linewidth=0.5)
    plt.show()
