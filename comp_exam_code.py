# %%

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import essentia

filename = 'sounds/Bad6.wav'
# import audio file
Fs, ystereo = wavfile.read(filename)

# consider only one channel
y1 = ystereo[:, 0]
y2 = ystereo[:, 1]
# length of the signal in samples
N = len(y1)
# time vector (in seconds)
t = np.divide(list(range(N)), Fs)

# removing DC component of the signal
y = y1 - np.mean(y1)
# absolute value of the audio signal
y_abs = np.absolute(y)

# ENVELOPE TRACKING
n_env = 250
# linear peak envelope vector initialization
local_maxima = np.zeros(int(N / n_env))
# linear envelope time vector initialization
time_ref = np.zeros(int(N / n_env))

for i in range(n_env, N+1, n_env):
    time_ref[int(i/n_env-1)] = i / Fs
    local_maxima[int(i/n_env-1)] = max(y_abs[i-n_env+1:i]
                                       )       # linear peak envelope

# print(N / n_env)

plt.plot(t, y_abs)
plt.plot(time_ref, local_maxima)
plt.show()

# minimum frequency considered for the pitch function
f_min = 150
# maximum frequency considered for the pitch function
f_max = 1000

# Given the interest in the transient, a short window with high overlap is chosen
window_length = round(Fs * 3 / f_min)
overlap_length = window_length // 10

f0 = librosa.yin(y, fmin=f_min, fmax=f_max, sr=Fs,
                 win_length=window_length, hop_length=overlap_length)

# pitch time vector (in samples)
idx_f0 = range(0, overlap_length * len(f0), overlap_length)
# pitch time vector (in seconds)
t_f0 = np.divide(idx_f0, Fs)

plt.figure()
plt.plot(t_f0, f0)
plt.show()

# ATTACK ESTIMATION/DETECTION ALGORITHM

amp_max = max(y_abs)            # maximum amplitude of the signal
# threshold to define the start attack time is a percentage of the maximum value of the amplitude envelope
thr_att = 0.15 * amp_max
# threshold to define the release time is a percentage of the maximum value of the amplitude envelope
thr_rel = 0.25 * amp_max

# Start of the attack definition
# The start of the attack is defined as the first sample in which the amplitude reaches the defined threshold
t_att_start = 0                 # attack start initialization

while y_abs[t_att_start] < thr_att:
    t_att_start += 1

t_att_start_sec = t_att_start / Fs          # attack start in seconds

# End of the attack definition
t_att_end = 0                   # attack end initialization

while idx_f0[t_att_end] < t_att_start:
    t_att_end += 1

# I then define the end of the attack as the point in which the fundamental
# frequency remains within a neighborhood of its value for at least t_min seconds
A4 = 442                        # reference pitch [Hz]

t_min = Fs * 0.3                # t_min definition
# frequency wide band within which a stable pitch is expected. 1 = 100 cents = 1 semitone
thr_f0 = 0.7

t_min_idx = 0
while idx_f0[t_min_idx] < t_min:
    t_min_idx += 1

while max(12 * np.log2(f0[t_att_end:t_att_end+t_min_idx+1]/A4)) - min((12 * np.log2(f0[t_att_end:t_att_end+t_min_idx+1]/A4))) > thr_f0:
    t_att_end += 1

t_att_end_sec = idx_f0[t_att_end] / Fs      # attack end in seconds

# Attack duration in milliseconds
att_dur = (t_att_end_sec - t_att_start_sec) * 10**3
print(att_dur)

# %%
