import numpy as np
from statistics import NormalDist
import pandas as pd
from progressbar import progressbar as bar
from itertools import product

from src.attack_time import get_attack_start_and_duration, AttackClarityError, get_spectrogram_from_file, get_attack_lenght

bad_sounds = [
    'sounds/Bad1.wav',
    'sounds/Bad2.wav',
    'sounds/Bad3.wav',
    # 'sounds/Bad4.wav',
    'sounds/Bad5.wav',
    'sounds/Bad6.wav',
    'sounds/Eb4_dirty1.wav',
    'sounds/Eb4_dirty2.wav',
    'sounds/Eb5_dirty1.wav',
    'sounds/Eb5_dirty2.wav',
    'sounds/Eb5_dirty3.wav',
]

good_sounds = [
    'sounds/Good1.wav',
    'sounds/Good2.wav',
    'sounds/Good3.wav',
    'sounds/Good4.wav',
    'sounds/Good5.wav',
    'sounds/Good6.wav',
    'sounds/Eb4_good.wav',
    'sounds/Eb5_good.wav',

]


def cost_function(good_values, bad_values):
    """
    this function calculates the distribution of the good and bad values
    - calculates the mean and std of the good and bad values
    - calculates the overlap of the gaussian distribution of the good and bad values
    - the cost is the overlap of the gaussian distribution of the good and bad values
    """

    min_bad = min(bad_values)
    max_good = max(good_values)

    # calculate the mean and std of the good and bad values
    good_mean = np.mean(good_values)
    good_std = np.std(good_values)
    bad_mean = np.mean(bad_values)
    bad_std = np.std(bad_values)

    # calculate the overlap of the gaussian distribution of the good and bad values
    if good_mean > bad_mean:
        return np.NaN, min_bad-max_good
    if good_std == 0 or bad_std == 0:
        return np.NaN, min_bad-max_good

    overlap = NormalDist(mu=good_mean, sigma=good_std).overlap(
        NormalDist(mu=bad_mean, sigma=bad_std))

    return overlap, min_bad-max_good


SPECTROGRAM_SAMPLE_TIME_MS = 1
spectrograms = {
    path: get_spectrogram_from_file(path, 44100, SPECTROGRAM_SAMPLE_TIME_MS=SPECTROGRAM_SAMPLE_TIME_MS) for path in bad_sounds + good_sounds
}


ATTACK_START_ENERGY = 0.3
NUMBER_OF_PEAKS = 5
ATTACK_PEAKS_ENERGY = 0.5
ROLLING_WINDOW = 30


# RANGE_ATTACK_START_ENERGY = np.linspace()
# RANGE_NUMBER_OF_PEAKS = np.linspace()
# RANGE_ATTACK_PEAKS_ENERGY = np.linspace()
# RANGE_ROLLING_WINDOW = np.linspace()
# RANGE_SPECTROGRAM_SAMPLE_TIME_MS = np.linspace()

attack_time_values = []
for attack_start_enery, \
        attack_peak_energy, \
        rolling_window, \
        number_of_peaks \
        in bar(list(product(
            np.linspace(0.2, 0.6, 8),
            np.linspace(0.3, 0.75, 8),
            [15, 20, 25, 30],
            [3, 4, 5, 6]
        ))):

    bad_values = []
    good_values = []

    values_dict = {}
    any_error = False

    for path in bad_sounds:
        try:
            start_time, end_time = get_attack_lenght(
                spectrograms[path],
                ATTACK_PEAKS_ENERGY=attack_peak_energy,
                NUMBER_OF_PEAKS=number_of_peaks,
                ATTACK_START_ENERGY=attack_start_enery,
                ROLLING_WINDOW=rolling_window
            )
            attack_time = end_time - start_time
            bad_values.append(attack_time)
            values_dict[path] = attack_time
        except AttackClarityError:
            any_error = True

    for path in good_sounds:
        try:
            start_time, end_time = get_attack_lenght(
                spectrograms[path],
                ATTACK_PEAKS_ENERGY=attack_peak_energy,
                NUMBER_OF_PEAKS=number_of_peaks,
                ATTACK_START_ENERGY=attack_start_enery,
                ROLLING_WINDOW=rolling_window
            )
            attack_time = end_time - start_time
            good_values.append(attack_time)
            values_dict[path] = attack_time
        except AttackClarityError:
            any_error = True

    row = {
        'attack_start_energy': attack_start_enery,
        'attack_peak_energy': attack_peak_energy,
        'rolling_window': rolling_window,
        'number_of_peaks': number_of_peaks,
        'any_error': 1 if any_error else 0,
        **values_dict
    }
    attack_time_values.append(row)
# print(attack_peak_energy, cost)

df = pd.DataFrame(attack_time_values)
df.to_excel('calibration.xlsx')
# df = df.set_index('attack_peak_energy')
# df.plot(ylim=(0, None))

df['cost'] = df.apply(lambda x: (cost_function(
    x[good_sounds], x[bad_sounds])[0]), axis=1)
