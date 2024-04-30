from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import librosa

ATTACK_START_ENERGY = 0.3
NUMBER_OF_PEAKS = 5
ATTACK_PEAKS_ENERGY = 0.5
ROLLING_WINDOW = 30
SPECTROGRAM_SAMPLE_TIME_MS = 1
# attack_max_ms = 250


class AttackClarityError(Exception):
    pass


def attack_time_from_file(path: str) -> tuple[float, float]:
    """
    Find the attack time of a played note from a file.

    Parameters
    ----------
    path : str
        The path to the file

    Returns
    -------
    tuple[float, float]
        The start and end time of the attack in milliseconds
    """
    sound_raw, sample_rate = librosa.load(path, sr=44100)
    return attack_time_from_array(sound_raw, sample_rate)


def attack_time_from_array(
        sound: np.ndarray,
        sample_rate: int
) -> tuple[float, float]:
    """
    Find the attack time of a played note.

    Parameters
    ----------
    sound : np.ndarray
        The sound to analyze
    sample_rate : int
        The sample rate of the sound

    Returns
    -------
    tuple[float, float]
        The start and end time of the attack in milliseconds
    """
    hop_length = sample_rate * SPECTROGRAM_SAMPLE_TIME_MS // 1000
    # attack_max_idx = attack_max_ms // spectogram_sample_rate_ms

    full_sp = librosa.feature.melspectrogram(
        y=sound,
        sr=sample_rate,
        hop_length=hop_length,
        n_mels=128,
        fmax=20000
    )

    energy = np.sum(full_sp, axis=0)
    mean_energy = np.mean(energy)

    over_threshold = np.where(energy > mean_energy * ATTACK_START_ENERGY)[0]

    if len(over_threshold) == 0:
        raise AttackClarityError('no energy above threshold')

    start_idx = over_threshold[0]
    # end_idx = start_idx + attack_max_idx

    sp = full_sp[:, start_idx:]
    attack_time = start_idx * SPECTROGRAM_SAMPLE_TIME_MS

    peak_energies = []
    for t in range(sp.shape[1]):
        s = sp[:, t]
        peaks = find_peaks(s)[0]
        peaks_values = s[peaks]

        # sort peaks by value
        order = np.argsort(peaks_values)[::-1]
        peaks = peaks[order]
        peaks_values = peaks_values[order]

        # take the first 5 peaks, sum the value and divide by total
        peaks_energy = np.sum(peaks_values[:NUMBER_OF_PEAKS]) / np.sum(s)
        peak_energies.append(peaks_energy)

    peaks_energy = np.array(peak_energies)
    time = np.arange(peaks_energy.shape[0]) * SPECTROGRAM_SAMPLE_TIME_MS

    serie = pd.Series(peaks_energy, index=time)
    df = pd.DataFrame(serie)
    df = df.rolling(window=ROLLING_WINDOW).min()

    # find the first peak that is above 0.5
    peaks_energy = df.values.flatten()
    over_threshold = np.where(peaks_energy > ATTACK_PEAKS_ENERGY)[0]

    if len(over_threshold) == 0:
        raise AttackClarityError(
            f'Attack Peaks Energy is too low (below {ATTACK_PEAKS_ENERGY})')

    first_peak = over_threshold[0]
    time_at_first_peak = time[first_peak]

    return attack_time, attack_time + time_at_first_peak


if __name__ == '__main__':
    paths = [
        'sounds/Bad1.wav',
        'sounds/Bad2.wav',
        'sounds/Bad3.wav',
        'sounds/Bad4.wav',
        'sounds/Bad5.wav',
        'sounds/Bad6.wav',
        'sounds/Good1.wav',
        'sounds/Good2.wav',
        'sounds/Good3.wav',
        'sounds/Good4.wav',
        'sounds/Good5.wav',
        'sounds/Good6.wav',
        'sounds/Eb4_dirty1.wav',
        'sounds/Eb4_dirty2.wav',
        'sounds/Eb4_good.wav',
        'sounds/Eb5_dirty1.wav',
        'sounds/Eb5_dirty2.wav',
        'sounds/Eb5_dirty3.wav',
        'sounds/Eb5_good.wav',
    ]
    for path in paths:
        sound_raw, sample_rate = librosa.load(path, sr=44100)
        try:
            start, end = attack_time_from_array(sound_raw, sample_rate)
            print(
                f'attack time {path}: {start} - {end}, duration: {end - start}ms'
            )
        except AttackClarityError as e:
            print(f'{path}: {e}')
