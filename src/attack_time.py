from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import librosa

DEFAULT_ATTACK_START_ENERGY = 0.3
DEFAULT_NUMBER_OF_PEAKS = 5
DEFAULT_ATTACK_PEAKS_ENERGY = 0.5
DEFAULT_ROLLING_WINDOW = 30
DEFAULT_SPECTROGRAM_SAMPLE_TIME_MS = 1
DEFAULT_PITCH_STABILITY_ROLLING_WINDOW_MS = 300
DEFAULT_PICH_STABILITY_THRESHOLD = 0.7
# attack_max_ms = 250
A4 = 442


class AttackClarityError(Exception):
    pass


def get_attack_start_and_duration(path: str, **kwargs) -> tuple[float, float]:
    """
    Find the attack time of a played note from a file.

    Parameters
    ----------
    path : str
        The path to the file
    ----------
    Keyword Arguments:
    ATTACK_START_ENERGY : float
        The energy threshold to consider the start of the attack
        default: 0.3
    NUMBER_OF_PEAKS : int
        The number of peaks to consider for the attack
        default: 5
    ATTACK_PEAKS_ENERGY : float
        The energy threshold to consider the attack peaks
        default: 0.5
    ROLLING_WINDOW : int
        The window size for the rolling minimum used to filter the peaks
        default: 30
    SPECTROGRAM_SAMPLE_TIME_MS : int
        The time in milliseconds to consider for the spectrogram
        default: 1

    Returns
    -------
    tuple[float, float]
        The start and end time of the attack in milliseconds
    """
    sample_time = kwargs.get(
        'SPECTROGRAM_SAMPLE_TIME_MS', DEFAULT_SPECTROGRAM_SAMPLE_TIME_MS)

    sound_raw, sample_rate = librosa.load(path, sr=44100)

    full_sp = get_spectrogram_from_file(path, sample_rate, **kwargs)

    attack_start_time_ms = get_attack_start_time(full_sp, **kwargs)

    sp_from_attack_start = full_sp[
        :, (attack_start_time_ms // sample_time):
    ]

    sound_from_attack_start = sound_raw[
        attack_start_time_ms * sample_rate//1000:
    ]

    df_peaks_energy = get_peaks_energy(
        sp_from_attack_start, sample_time, **kwargs)
    df_pitch = get_pitch(sound_from_attack_start,
                         sample_rate, sample_time, **kwargs)

    attack_duration_ms = get_attack_time(df_pitch, df_peaks_energy, **kwargs)

    return attack_start_time_ms, attack_duration_ms


def get_attack_start_time(full_sp, **kwargs) -> float:
    ATTACK_START_ENERGY = kwargs.get(
        'ATTACK_START_ENERGY', DEFAULT_ATTACK_START_ENERGY)

    SPECTROGRAM_SAMPLE_TIME_MS = kwargs.get(
        'SPECTROGRAM_SAMPLE_TIME_MS', DEFAULT_SPECTROGRAM_SAMPLE_TIME_MS)

    energy = np.sum(full_sp, axis=0)
    mean_energy = np.mean(energy)

    over_threshold = np.where(energy > mean_energy * ATTACK_START_ENERGY)[0]

    if len(over_threshold) == 0:
        raise AttackClarityError('no energy above threshold')

    start_idx = over_threshold[0]
    attack_time = start_idx * SPECTROGRAM_SAMPLE_TIME_MS

    return attack_time


def get_attack_time(df_pitch: pd.DataFrame, df_peaks_energy: pd.DataFrame, **kwargs) -> int:
    ATTACK_PEAKS_ENERGY = kwargs.get(
        'ATTACK_PEAKS_ENERGY', DEFAULT_ATTACK_PEAKS_ENERGY)
    ROLLING_WINDOW = kwargs.get('ROLLING_WINDOW', DEFAULT_ROLLING_WINDOW)

    PITCH_STABILITY_ROLLING_WINDOW_MS = kwargs.get(
        'PITCH_STABILITY_ROLLING_WINDOW_MS', DEFAULT_PITCH_STABILITY_ROLLING_WINDOW_MS)
    PICH_STABILITY_THRESHOLD = kwargs.get(
        'PICH_STABILITY_THRESHOLD', DEFAULT_PICH_STABILITY_THRESHOLD)

    sample_time = kwargs.get(
        'SPECTROGRAM_SAMPLE_TIME_MS', DEFAULT_SPECTROGRAM_SAMPLE_TIME_MS)

    pitch_stability_window = PITCH_STABILITY_ROLLING_WINDOW_MS // sample_time
    pitch_stability = df_pitch['pitch_note'].rolling(pitch_stability_window).apply(
        lambda vals: vals.max() - vals.min() < PICH_STABILITY_THRESHOLD).shift(-pitch_stability_window)

    peaks_energy = df_peaks_energy\
        .rolling(window=ROLLING_WINDOW).min()

    time = peaks_energy.index

    # find the first peak that is above 0.5
    peaks_energy_stable = peaks_energy.values.flatten() > ATTACK_PEAKS_ENERGY
    pitch_stable = pitch_stability.values.flatten() > 0

    # make the two arrays the same length
    if len(peaks_energy_stable) > len(pitch_stable):
        peaks_energy_stable = peaks_energy_stable[:len(pitch_stable)]
    elif len(pitch_stable) > len(peaks_energy_stable):
        pitch_stable = pitch_stable[:len(peaks_energy_stable)]

    over_threshold = np.where((peaks_energy_stable & pitch_stable))[0]

    if len(over_threshold) == 0:
        raise AttackClarityError(
            f'Attack Peaks Energy is too low (below {ATTACK_PEAKS_ENERGY})')

    first_peak = over_threshold[0]
    time_at_first_peak = time[first_peak]

    return time_at_first_peak


def get_peaks_energy(sp, sample_time, **kwargs) -> pd.DataFrame:
    NUMBER_OF_PEAKS = kwargs.get('NUMBER_OF_PEAKS', DEFAULT_NUMBER_OF_PEAKS)

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
    time = np.arange(peaks_energy.shape[0]) * sample_time

    serie = pd.Series(peaks_energy, index=time)
    df = pd.DataFrame(serie)

    return df


def get_pitch(sound_raw, sample_rate, sample_time, **kwargs) -> pd.DataFrame:
    hop_length = sample_rate * sample_time // 1000
    pitch_freq = librosa.yin(
        sound_raw,
        fmin=150,
        fmax=1000,
        sr=sample_rate,
        hop_length=hop_length
    )

    pitch_note = 12 * np.log2(pitch_freq/A4)
    df_pitch = pd.DataFrame(
        pitch_note,
        index=np.arange(pitch_note.shape[0]) * sample_time,
        columns=['pitch_note']
    )

    return df_pitch


def get_spectrogram_from_file(path: str, sample_rate, **kwargs):
    sound_raw, sample_rate = librosa.load(path, sr=sample_rate)

    SPECTROGRAM_SAMPLE_TIME_MS = kwargs.get(
        'SPECTROGRAM_SAMPLE_TIME_MS',
        DEFAULT_SPECTROGRAM_SAMPLE_TIME_MS
    )

    hop_length = sample_rate * SPECTROGRAM_SAMPLE_TIME_MS // 1000
    # attack_max_idx = attack_max_ms // spectogram_sample_rate_ms

    full_sp = librosa.feature.melspectrogram(
        y=sound_raw,
        sr=sample_rate,
        hop_length=hop_length,
        n_mels=128,
        fmax=20000
    )
    return full_sp


if __name__ == '__main__':
    paths = [
        'sounds/Good1.wav',
        'sounds/Eb4_dirty1.wav',
        'sounds/Eb4_dirty2.wav',
        'sounds/Eb5_dirty1.wav',
        'sounds/Eb5_dirty2.wav',
        'sounds/Eb5_dirty3.wav',
        'sounds/Bad1.wav',
        'sounds/Bad2.wav',
        'sounds/Bad3.wav',
        # 'sounds/Bad4.wav',
        'sounds/Bad5.wav',
        'sounds/Bad6.wav',

        'sounds/Eb4_good.wav',
        'sounds/Eb5_good.wav',

        'sounds/Good1.wav',
        'sounds/Good2.wav',
        'sounds/Good3.wav',
        'sounds/Good4.wav',
        'sounds/Good5.wav',
        'sounds/Good6.wav',

    ]
    for path in paths:

        try:
            start, duration = get_attack_start_and_duration(path, sr=44100)
            print(
                f'{path} -> start: {start}ms, duration: {duration}ms'
            )
        except AttackClarityError as e:
            print(f'{path}: {e}')
