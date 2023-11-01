import csv
import numpy as np
import matplotlib.pyplot as plt


x_data = []
y_data = []

output_dir = "continouse_dataset/"
file_path = output_dir + 'loss_history.csv'

with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

z = np.polyfit(x_data, y_data, 1)
p = np.poly1d(z)
x_intersection = -z[1] / z[0]
print("x_intersection : " + str(x_intersection))
plt.plot(x_data, y_data, label='Loss')
plt.plot(x_data, p(x_data), label='Trendline')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.savefig(output_dir + 'loss_graph.png')
