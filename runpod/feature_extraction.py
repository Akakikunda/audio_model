import librosa
import numpy as np

def extract_tabular(y, sr):
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f'mfcc{i+1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc{i+1}_std'] = float(np.std(mfcc[i]))
    for name, fn in [
        ('spectral_centroid', lambda: librosa.feature.spectral_centroid(y=y, sr=sr)),
        ('spectral_bandwidth', lambda: librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        ('spectral_rolloff', lambda: librosa.feature.spectral_rolloff(y=y, sr=sr)),
        ('zcr', lambda: librosa.feature.zero_crossing_rate(y)),
        ('chroma', lambda: librosa.feature.chroma_stft(y=y, sr=sr)),
        ('rms', lambda: librosa.feature.rms(y=y)),
    ]:
        v = fn()
        feats[f'{name}_mean'] = float(np.mean(v))
        feats[f'{name}_std'] = float(np.std(v))
    return feats

