import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#WALKING DATA
# Read the datasets
data1 = pd.read_csv('./ryan walking/ryan walking.csv')
data2 = pd.read_csv('./alan walking/alan walking.csv')
data3 = pd.read_csv('./kai walking/kai walking.csv')

# Initialize a figure for 3D plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Plot data from the first dataset
ax.scatter(data1['Acceleration x (m/s^2)'], data1['Acceleration y (m/s^2)'], data1['Acceleration z (m/s^2)'], color='red', label='Ryan Walking Data')

# Plot data from the second dataset
ax.scatter(data2['Acceleration x (m/s^2)'], data2['Acceleration y (m/s^2)'], data2['Acceleration z (m/s^2)'], color='blue', label='Alan Walking Data')

# Plot data from the third dataset
ax.scatter(data3['Acceleration x (m/s^2)'], data3['Acceleration y (m/s^2)'], data3['Acceleration z (m/s^2)'], color='green', label='Kai Walking Data')

# Adding legend to differentiate the datasets
ax.legend()

# Adding title and labels
ax.set_title('Visualization of Individual Walking Acceleration')
ax.set_xlabel('Acceleration x (m/s^2)')
ax.set_ylabel('Acceleration y (m/s^2)')
ax.set_zlabel('Acceleration z (m/s^2)')  # Correct way to set the label for the z-axis

# Show the plot
plt.show()




#JUMPING DATA
# Read the datasets
data1 = pd.read_csv('./ryan jumping/ryan jumping.csv')
data2 = pd.read_csv('./alan jumping/alan jumping.csv')
data3 = pd.read_csv('./kai jumping/kai jumping.csv')

# Initialize a figure for 3D plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Plot data from the first dataset
ax.scatter(data1['Acceleration x (m/s^2)'], data1['Acceleration y (m/s^2)'], data1['Acceleration z (m/s^2)'], color='purple', label='Ryan Jumping Data')

# Plot data from the second dataset
ax.scatter(data2['Acceleration x (m/s^2)'], data2['Acceleration y (m/s^2)'], data2['Acceleration z (m/s^2)'], color='orange', label='Alan Jumping Data')

# Plot data from the third dataset
ax.scatter(data3['Acceleration x (m/s^2)'], data3['Acceleration y (m/s^2)'], data3['Acceleration z (m/s^2)'], color='brown', label='Kai Jumping Data')

# Adding legend to differentiate the datasets
ax.legend()

# Adding title and labels
ax.set_title('Visualization of Individual Jumping Acceleration')
ax.set_xlabel('Acceleration x (m/s^2)')
ax.set_ylabel('Acceleration y (m/s^2)')
ax.set_zlabel('Acceleration z (m/s^2)')  # Correct way to set the label for the z-axis

# Show the plot
plt.show()




def readLineData(inputFile, activity, participant, color):
    # Reading the data from the csv file
    points = pd.read_csv(inputFile)

    # Extracting Time and Absolute acceleration values
    x = points['Time (s)'].values
    y = points['Absolute acceleration (m/s^2)'].values

    # Plotting the data
    plt.figure(figsize=(10, 6))  # Set the figure size for better visibility
    plt.plot(x, y, label=f'{participant} {activity}', color=color)  # Plot with specified color and label for legend

    # Setting the title and labels
    plt.title(f'{participant} - {activity}')  # Title to differentiate the activities and participants
    plt.xlabel('Time (s)')  # X-axis label
    plt.ylabel('Absolute acceleration (m/s^2)')  # Y-axis label

    plt.legend()  # Display legend to identify the plots
    plt.show()  # Show the plot

# Example calls to the function with color specified for each plot
readLineData('./ryan walking/ryan walking.csv', 'Walking', 'Ryan', 'blue')
readLineData('./alan walking/alan walking.csv', 'Walking', 'Alan', 'green')
readLineData('./kai walking/kai walking.csv', 'Walking', 'Kai', 'red')
readLineData('./ryan jumping/ryan jumping.csv', 'Jumping', 'Ryan', 'cyan')
readLineData('./alan jumping/alan jumping.csv', 'Jumping', 'Alan', 'magenta')
readLineData('./kai jumping/kai jumping.csv', 'Jumping', 'Kai', 'yellow')












#META DATA for phone positions
y = np.array([16.67, 16.67, 16.67, 16.67, 16.67, 16.67])

# Labels for each section
labels = ['Walking with Phone in Right Pocket', 'Walking with Phone in Arm Band', 'Walking with Phone in Left Hand', 'Jumping with Phone in Right Pocket', 'Jumping with Phone in Arm Band', 'Jumping with Phone in Left Hand']

# Adding a title
plt.title('Meta Data: Distribution of Phone Positions During Walking and Jumping')

# Creating the pie chart with labels and displaying the percentage value
plt.pie(y, labels=labels, autopct='%1.1f%%')

# Display the plot
plt.show()






#META DATA for phone types
y = np.array([66.66, 33.33])

# Labels for each section
labels = ['iPhone 13', 'iPhone 11 Pro']

# Adding a title
plt.title('Meta Data: iPhone Type Used for Data Collection')

# Creating the pie chart with labels and displaying the percentage value
plt.pie(y, labels=labels, autopct='%1.1f%%')

# Display the plot
plt.show()




