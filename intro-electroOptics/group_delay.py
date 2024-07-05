import numpy as np
import sounddevice as sd
import time

# Define the sample rate and duration of the sound
sample_rate = 44100  # 44.1 kHz
duration = 2  # Duration of the sound in seconds

# Define the frequencies of the sine waves
frequency1 = 400    # frequency in Hz of the first  sine wave
frequency2 = 403    # frequency in Hz of the second sine wave
# powerDips  = 3    # number of power dips per second
# frequency2 = frequency1 + powerDips

# Generate the time axis
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate the sine waves
wave1 = np.sin(2 * np.pi * frequency1 * t)
wave2 = np.sin(2 * np.pi * frequency2 * t)

# Add the two sine waves together
result = wave1 + wave2

# Play the resulting sound
sd.play(result, sample_rate)
sd.wait()
