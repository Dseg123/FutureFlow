import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from matplotlib.animation import FuncAnimation

mat = scipy.io.loadmat('csat_data_5_27_2022.mat')

u_vals = mat['u']
v_vals = mat['v']
w_vals = mat['w']
u_vals = np.array(u_vals)
v_vals = np.array(v_vals)
w_vals = np.array(w_vals)

# Create a figure and axis
fig, ax = plt.subplots()

# Create an empty scatter plot (initially no point)
plot, = ax.plot([], [])
# arrow = ax.arrow(0, 0, 0, 0)
#scatter, = ax.plot([], [], 'o')

# Set the axis limits
ax.set_xlabel("u (m/s)")
ax.set_ylabel("v (m/s)")
ax.set_title("Horizontal wind vector on 5/27/22")
ax.set_xlim(-5, 5)  # Adjust these limits as needed
ax.set_ylim(-5, 5)  # Adjust these limits as needed

def update(frame):
    print(frame)
    # Calculate new x and y values based on your x[t] and y[t] data
    x_value = u_vals[frame]
    y_value = v_vals[frame]

    # Update the data for the scatter plot
    #scatter.set_data([0,x_value], [0,y_value])
    plot.set_data([0, x_value], [0, y_value])
    # arrow.set_data(x=0, y=0, dx=x_value, dy=y_value)

    return plot,

# Replace `num_frames` with the total number of frames in your animation
num_frames = len(u_vals)
print(num_frames)

# Create the animation
animation = FuncAnimation(fig, update, frames=1000, blit=True, interval=10)

# Display the animation (this will show a window with the animation)
#plt.show()

animation.save('vane.mp4', writer='ffmpeg', fps=10)