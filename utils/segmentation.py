from pydub import AudioSegment
import numpy as np


def segment_utterance(utterance_path):
    audio = AudioSegment.from_wav(utterance_path)
    segments = Window(audio)
    return segments


def Window(signal, w_size=500):
    segments = []
    start = 0
    while start < len(signal):
        end = start+w_size
        print(start)
        print(end)
        if end <= len(signal):
            sound = signal[start:end]
            sound = np.asarray(sound.get_array_of_samples(),dtype = np.int64)
            max_amp = max(sound)
            segments.append(sound / max_amp)
            start += (w_size/2)
        else:
            break
    return segments
