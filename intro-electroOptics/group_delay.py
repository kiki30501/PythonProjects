
######################################################################################
# Date:         05/07/24
# Last Update:  05/07/24
# Course:       Intro to Electro-Optics
# Author:       Reshef Schachter
# Project:      Demonstration of the group delay effect
# Description:  Using the computer speaker to demonstrate the group delay effect.
#               The number of "wubs" per second is the difference between the two
#               frequencies. As it's set now, the difference is 3 "wubs" per second.
#
# "wubs" is a term I made up to describe the effect of the two frequencies interfering
# with each other. I called it that because that's what it sounds like to me.
######################################################################################


import numpy as np
import sounddevice as sd    # Read the technical note below
import time

##################################### Technical Note ########################################
# The "sounddevice" library (and also the numpy library) is not installed by default. You can
# install it using pip. write the following function in the terminal:
# 
#   pip install sounddevice
#   (if you need numpy, write "pip install numpy" once you're done with sounddevice)
# 
# if the installation succeeded, you should see the following message (or something similar):
# "Successfully installed CFFI-1.16.0 pycparser-2.22 sounddevice-0.4.7"
# if you see an error.... well, just ask google what to do.
# 
# All of this is assuming you have pip installed. If you don't, then google how to install it.
#############################################################################################



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
