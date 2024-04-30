# Attack time detection for recorded sounds

This project is a simple implementation of a sound attack time detection algorithm. The algorithm. The algorithm is implemented in Python and uses the librosa library for audio processing.

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