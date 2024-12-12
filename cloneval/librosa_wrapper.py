import numpy as np
from typing import Union
import scipy.stats
import torch
import librosa
import scipy

class LibrosaWrapper:
    """
    A wrapper class to apply various Librosa transformations to audio data.

    Args:
        sampling_rate (int): The sampling rate of the input audio. Default is 16000.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 2048.
        hop_length (int): Length of hop between STFT windows. Default is 512.
        n_mfcc (int): Number of MFCCs to retain. Default is 13.
    """
    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
    ):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc

    def __call__(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int) -> dict:
        """
        Process the input waveform through various Librosa transformations.

        Args:
            waveform (Union[torch.Tensor, np.ndarray, list]): The input audio waveform.
            sampling_rate (int): The sampling rate of the input audio.

        Returns:
            dict: The processed audio features.
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        elif isinstance(waveform, list):
            waveform = np.array(waveform)
        
        if sampling_rate != self.sampling_rate:
            waveform = librosa.resample(y=waveform, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        
        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sampling_rate)

        specgram = np.abs(librosa.stft(y=waveform, hop_length=self.hop_length, n_fft=self.n_fft))
        specgram_delta1 = librosa.feature.delta(specgram)

        mel_specgram = librosa.feature.melspectrogram(S=specgram, sr=self.sampling_rate)
        mel_specgram_delta1 = librosa.feature.delta(mel_specgram)

        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
        mfcc_delta1 = librosa.feature.delta(mfcc)

        fmt_ac = librosa.autocorrelate(y=onset_env, max_size=10 * self.sampling_rate // self.hop_length)
        fmt_ac = librosa.util.normalize(fmt_ac, norm=np.inf)
        
        return {
            'pitch': librosa.pyin(y=waveform, sr=self.sampling_rate, fmin=65, fmax=2093)[0],
            'spectrogram': specgram,
            'spectrogram_delta1': specgram_delta1,
            'mel_spectrogram': librosa.power_to_db(mel_specgram, ref=np.max),
            'mel_spectrogram_delta1': mel_specgram_delta1,
            'mfcc': mfcc,
            'mfcc_delta1': mfcc_delta1,
            'mfcc_delta2': librosa.feature.delta(mfcc_delta1),
            'rms': librosa.feature.rms(y=waveform),
            'spectral_centroid': librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
            'spectral_contrast': librosa.feature.spectral_contrast(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
            'spectral_flatness': librosa.feature.spectral_flatness(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length),
            'spectral_rollof': librosa.feature.spectral_rolloff(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=waveform, frame_length=self.n_fft, hop_length=self.hop_length),
            'static_tempo': librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate),
            'static_tempo_uniform': librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, prior=scipy.stats.uniform(30, 300)),
            'dynamic_tempo': librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, aggregate=None),
            'dynamic_tempo_lognorm': librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, aggregate=None, prior=scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)),
            'linear_prediction_coefficients': librosa.lpc(y=waveform, order=2),
            'tempogram': librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sampling_rate, hop_length=self.hop_length),
            'chromagram': librosa.feature.chroma_stft(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
            'pseudo_const_Q_transform': np.abs(librosa.pseudo_cqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length, fmin=65, n_bins=60 * 2, bins_per_octave=12 * 2)),
            'iirt': librosa.amplitude_to_db(np.abs(librosa.iirt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
            'variable_Q_transform': librosa.amplitude_to_db(np.abs(librosa.vqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
            'const_Q_chromagram': librosa.feature.chroma_cqt(y=waveform, sr=self.sampling_rate),
        }
    

    def get_feature_list(self):
        feature_list = [
        'pitch',
        'spectrogram',
        'spectrogram_delta1',
        'mel_spectrogram',
        'mel_spectrogram_delta1',
        'mfcc',
        'mfcc_delta1',
        'mfcc_delta2',
        'rms',
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_contrast',
        'spectral_flatness',
        'spectral_rollof',
        'zero_crossing_rate',
        'static_tempo',
        'static_tempo_uniform',
        'dynamic_tempo',
        'dynamic_tempo_lognorm',
        'linear_prediction_coefficients',
        'tempogram',
        'chromagram',
        'pseudo_const_Q_transform',
        'iirt',
        'variable_Q_transform',
        'const_Q_chromagram']

        return feature_list