import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(100)
def von_karman_spectrum(frequency, L, sigma_v):
    return sigma_v**2 * (0.5 * L / (2 * np.pi)) / (1 + (frequency * L / (2 * np.pi))**2)**(5 / 6)

def generate_wind_data_3d(num_samples, time_step, sigma_v, L):
    frequencies = np.fft.fftfreq(num_samples, time_step)
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(3, len(frequencies)))

    amplitude = np.sqrt(np.array([von_karman_spectrum(f, L, sigma_v) for f in frequencies]))
    amplitude[0] = 0  
    fft_data = np.zeros((3, num_samples))
    fft_data[0, :] = amplitude * random_phases[0, :]
    fft_data[1, :] = amplitude * random_phases[1, :]
    fft_data[2, :] = amplitude * random_phases[2, :]

    wind_data_3d = np.fft.ifft(fft_data, axis=1).real
    return wind_data_3d


num_samples = 10 * 50000  # Number of time samples
time_step = 0.1  # Time step between samples (in seconds)
sigma_v = 1000.0  # Velocity standard deviation
L = 100  # Length scale of turbulence

# Generate 3D wind data
wind_data_3d = generate_wind_data_3d(num_samples, time_step, sigma_v, L)
print(wind_data_3d)

w3 = wind_data_3d
size = 100 #num_samples
real_df = pd.read_csv('sample_data.csv')
std_u = real_df.iloc[:size]['u'].std() / (w3[:size, 0].std())
mean_u = real_df.iloc[:size]['u'].mean()
std_v = real_df.iloc[:size]['v'].std() / (w3[:size, 1].std())
mean_v = real_df.iloc[:size]['v'].mean()
std_w = real_df.iloc[:size]['w'].std() / (w3[:size, 2].std())
mean_w = real_df.iloc[:size]['w'].mean()
print(std_u, std_v, std_w)

d = {'u': std_u * wind_data_3d[0, :] + mean_u, 'v': std_v * wind_data_3d[1, :] + mean_v, 'w': std_w * wind_data_3d[2, :] + mean_w}
df = pd.DataFrame.from_dict(d)
# df.to_csv('von_karman_data_2.csv')

# Plotting the generated wind data for each component
time = np.arange(0, num_samples * time_step, time_step)
components = ['u', 'v', 'w']
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(wind_data_3d[i][:100].real, label=components[i])

plt.title('3D Synthetic Wind Data (Von Karman Turbulence Model)')
plt.xlabel('Time (seconds)')
plt.ylabel('Wind Speed')
plt.legend()
plt.grid(True)
plt.show()