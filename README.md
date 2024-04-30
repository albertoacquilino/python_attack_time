# Attack time detection for recorded sounds

This project is a simple implementation of a sound attack time detection algorithm. 

The algorithm is implemented in Python and uses the librosa library for audio processing.

## Installation
Clone the repository and install the required libraries using pip:
```bash
git clone
pip install -r requirements.txt
```

or
    
```bash
python setup.py install
```

## Usage
The algorithm is implemented in the `attack_time.py` file. 
The attack_time module implements two functions:
- `attack_time_from_file` which will return the attack time and duration of a sound file in milliseconds
- `attack_time_from_array` which will return the attack time and duration of a sound array in milliseconds

```python
from attack_time import attack_time_from_file, attack_time_from_array


# Attack time from file
attack_start, attack_end = attack_time_from_file('path/to/file.wav')
print(f'Attack start: {attack_start}ms, Attack end: {attack_end}ms, duration: {attack_end - attack_start}ms')

# otherwise load the file and pass the array to the attack_time_from_array function
import librosa
y, sr = librosa.load('path/to/file.wav')
attack_start, attack_end = attack_time_from_array(y, sr)
print(f'Attack start: {attack_start}ms, Attack end: {attack_end}ms, duration: {attack_end - attack_start}ms')

```

## Hyperparameters
The algorithm uses the following hyperparameters that can be overridden as kwargs:
- **ATTACK_START_ENERGY** : float
        The energy threshold to consider the start of the attack
        default: 0.3
- **NUMBER_OF_PEAKS** : int
        The number of peaks to consider for the attack
        default: 5
- **ATTACK_PEAKS_ENERGY** : float
        The energy threshold to consider the attack peaks
        default: 0.5
- **ROLLING_WINDOW** : int
        The window size for the rolling minimum used to filter the peaks
        default: 30
- **SPECTROGRAM_SAMPLE_TIME_MS** : int
        The time in milliseconds to consider for the spectrogram
        default: 1
