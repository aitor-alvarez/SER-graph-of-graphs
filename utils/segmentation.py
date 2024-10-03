from pydub import AudioSegment

def segment_utterance(utterance_path):
    audio = AudioSegment.from_wav(utterance_path)
    windowing(audio)
    return None

def windowing(signal, w_size=500):
    start = 0
    while start < len(signal):
        end = start+w_size
        if end <= len(signal):
            sound_sample = signal[start:end]
            sound_sample.export("tmp/"+str(start)+".mp3",format="mp3", bitrate="192k")
            start += (w_size/2)
        else:
            break
    return None