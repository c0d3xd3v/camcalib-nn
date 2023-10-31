import csv
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the necessary libraries

# Step 2: Read the data from the CSV file
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
plt.plot(x_data, y_data, label='Loss')  # Create a simple line plot
plt.plot(x_data, p(x_data), label='Trendline')
#lt.title('CSV Data Plot')  # Set the title of the plot
plt.xlabel('epoch')  # Label for the x-axis
plt.ylabel('loss')  # Label for the y-axis
plt.legend()  # Display the legend (if multiple datasets)

# Step 5: Display the plot
plt.show()
