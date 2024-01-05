# import numpy as np
# import matplotlib.pyplot as plt

# def dryden_spectrum(frequency, airspeed, length_scale):
#     """
#     Dryden spectrum calculation
#     """
#     return (4 * length_scale * airspeed**2 * frequency) / (1 + (length_scale * frequency)**2)**(5 / 6)


# def von_karman_spectrum(frequency, L, sigma_v):
#     """
#     Von Karman spectrum calculation
#     """
#     return sigma_v**2 * (0.5 * L / (2 * np.pi)) / (1 + (frequency * L / (2 * np.pi))**2)**(5 / 6)

# def generate_wind_data(num_samples, time_step, sigma_v, L):
#     """
#     Generate time series wind data using Von Karman turbulence model
#     """
#     frequencies = np.fft.fftfreq(num_samples, time_step)
#     random_phases = np.exp(1j * 2 * np.pi * np.random.rand(len(frequencies)))

#     amplitude = np.sqrt(np.array([von_karman_spectrum(f, L, sigma_v) for f in frequencies]))
#     amplitude[0] = 0  # DC component is set to 0

#     # Generate random complex numbers in frequency domain
#     fft_data = amplitude * random_phases

#     # Inverse FFT to get time series wind data
#     wind_data = np.fft.ifft(fft_data).real
#     return wind_data

# # Parameters
# num_samples = 10 * 500000  # Number of time samples
# time_step = 0.1  # Time step between samples (in seconds)
# sigma_v = 50.0  # Velocity standard deviation
# L = 100  # Length scale of turbulence

# # Generate wind data
# wind_data = generate_wind_data(num_samples, time_step, sigma_v, L)
# print(wind_data)
# # # Plotting the generated wind data
# # time = np.arange(0, num_samples * time_step, time_step)
# # plt.figure(figsize=(10, 6))
# # plt.plot(time, wind_data)
# # plt.title('Synthetic Wind Data (Von Karman Turbulence Model)')
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Wind Speed')
# # plt.grid(True)
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def von_karman_spectrum(frequency, L, sigma_v):
    """
    Von Karman spectrum calculation
    """
    return sigma_v**2 * (0.5 * L / (2 * np.pi)) / (1 + (frequency * L / (2 * np.pi))**2)**(5 / 6)

def generate_wind_data_3d(num_samples, time_step, sigma_v, L):
    """
    Generate 3D wind data using Von Karman turbulence model
    """
    frequencies = np.fft.fftfreq(num_samples, time_step)
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(3, len(frequencies)))

    amplitude = np.sqrt(np.array([von_karman_spectrum(f, L, sigma_v) for f in frequencies]))
    amplitude[0] = 0  # DC component is set to 0

    # Generate random complex numbers in frequency domain for each component
    fft_data = np.zeros((3, num_samples))
    fft_data[0, :] = amplitude * random_phases[0, :]
    fft_data[1, :] = amplitude * random_phases[1, :]
    fft_data[2, :] = amplitude * random_phases[2, :]

    # Inverse FFT to get time series wind data
    wind_data_3d = np.fft.ifft(fft_data, axis=1).real
    return wind_data_3d

# Parameters
num_samples = 10 * 50000  # Number of time samples
time_step = 0.1  # Time step between samples (in seconds)
sigma_v = 1000.0  # Velocity standard deviation
L = 100  # Length scale of turbulence

# Generate 3D wind data
wind_data_3d = generate_wind_data_3d(num_samples, time_step, sigma_v, L)
print(wind_data_3d)

d = {'u': wind_data_3d[0, :], 'v': wind_data_3d[1, :], 'w': wind_data_3d[2, :]}
df = pd.DataFrame.from_dict(d)
df.to_csv('von_karman_data.csv')

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