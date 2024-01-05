import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from matplotlib.animation import FuncAnimation

name = "Optimal"
df = pd.read_csv(f"{name}_sim_results.csv")
x = df['x']

# Create a figure and axis
fig, ax = plt.subplots()

# Create an empty scatter plot (initially no point)
plot, = ax.plot([], [], 'o')
plot2, = ax.plot([], [], 'x')
# arrow = ax.arrow(0, 0, 0, 0)
#scatter, = ax.plot([], [], 'o')

# Set the axis limits
ax.set_xlabel("Displacement (m)")
ax.set_title(f"{name} Model - Simulation Results")
ax.set_xlim(-.2, .2)  # Adjust these limits as needed
ax.set_ylim(-.5, .5)  # Adjust these limits as needed

def update(frame):
    print(frame)
    # Calculate new x and y values based on your x[t] and y[t] data
    x_value = x.iloc[frame]

    # Update the data for the scatter plot
    # scatter.set_data([0,x_value], [0,y_value])
    plot.set_data([x_value], [0])
    plot2.set_data([0], [0])
    # arrow.set_data(x=0, y=0, dx=x_value, dy=y_value)

    return plot,plot2

# Replace `num_frames` with the total number of frames in your animation
num_frames = len(x)
print(num_frames)

# Create the animation
animation = FuncAnimation(fig, update, frames=100, blit=True, interval=10)

# Display the animation (this will show a window with the animation)
#plt.show()

animation.save(f'{name}.mp4', writer='ffmpeg', fps=10)