from pydub import AudioSegment


def segment_utterance(utterance_path):
    audio = AudioSegment.from_wav(utterance_path)
    segments = Window(audio)
    return segments


def Window(signal, w_size=500):
    segments = []
    start = 0
    while start < len(signal):
        end = start+w_size
        if end <= len(signal):
            segments.append(signal[start:end])
            start += (w_size/2)
        else:
            continue
    return segments
