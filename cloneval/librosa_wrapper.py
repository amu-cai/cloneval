import numpy as np
from typing import Union
import torch
import librosa


class LibrosaWrapper:
    """
    A wrapper class to apply various Librosa transformations to audio data.

    Args:
        sampling_rate (int): The sampling rate of the input audio. Default is 16000.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 2048.
        hop_length (int): Length of hop between STFT windows. Default is 512.
        n_mfcc (int): Number of MFCCs to retain. Default is 13.
    """
    def __init__(self, sampling_rate: int = 16_000, n_fft: int = 2048, hop_length: int = 512):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

    def pitch(self, waveform: np.ndarray, fmin: int = 65, fmax: int = 2093) -> np.ndarray:
        return librosa.pyin(y=waveform, sr=self.sampling_rate, fmin=fmin, fmax=fmax)[0]
    
    def spectrogram_mel_spectrogram(self, waveform: np.ndarray) -> dict:
        stft = librosa.stft(y=waveform, hop_length=self.hop_length, n_fft=self.n_fft)
        spectrogram = np.abs(stft)
        mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, sr=self.sampling_rate)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return {
            "spectrogram": spectrogram, 
            "mel_spectrogram": mel_spectrogram,
        }
    
    def mfccs(self, waveform: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        return librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=n_mfcc)
    
    def rms(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.rms(y=waveform)
    
    def spectral_centroid(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)

    def spectral_bandwidth(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.spectral_bandwidth(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)

    def spectral_contrast(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.spectral_contrast(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)

    def spectral_flatness(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.spectral_flatness(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length)

    def spectral_rolloff(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.spectral_rolloff(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)
    
    def zero_crossing_rate(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.zero_crossing_rate(y=waveform, frame_length=self.n_fft, hop_length=self.hop_length)
    
    def lpcs(self, waveform: np.ndarray, order: int = 2) -> np.ndarray:
        return librosa.lpc(y=waveform, order=order)
    
    def tempogram(self, waveform: np.ndarray) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(y=waveform, sr=self.sampling_rate)
        return librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sampling_rate, hop_length=self.hop_length)
    
    def chromagram(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.chroma_stft(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)
    
    def pseudo_const_Q_transform(self, waveform: np.ndarray, fmin: int = 65, n_bins: int = 120, bins_per_octave: int = 24) -> np.ndarray:
        pseudo_cqt = librosa.pseudo_cqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
        return np.abs(pseudo_cqt)

    def iirt(self, waveform: np.ndarray) -> np.ndarray:
        iirt = librosa.iirt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)
        iirt = np.abs(iirt)
        return librosa.amplitude_to_db(iirt, ref=np.max)

    def variable_Q_transform(self, waveform: np.ndarray) -> np.ndarray:
        vqt = librosa.vqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)
        vqt = np.abs(vqt)
        return librosa.amplitude_to_db(vqt, ref=np.max)

    def const_Q_chromagram(self, waveform: np.ndarray) -> np.ndarray:
        return librosa.feature.chroma_cqt(y=waveform, sr=self.sampling_rate)
    
    def preprocess(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int) -> np.ndarray:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        elif isinstance(waveform, list):
            waveform = np.array(waveform)
        
        if sampling_rate != self.sampling_rate:
            waveform = librosa.resample(y=waveform, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        
        return waveform
    
    def __call__(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int) -> dict:
        waveform = self.preprocess(waveform, sampling_rate)

        features = {
            "pitch": self.pitch(waveform),
            "mfccs": self.mfccs(waveform),
            "rms": self.rms(waveform),
            "spectral_centroid": self.spectral_centroid(waveform),
            "spectral_bandwidth": self.spectral_bandwidth(waveform),
            "spectral_contrast": self.spectral_contrast(waveform),
            "spectral_flatness": self.spectral_flatness(waveform),
            "spectral_rolloff": self.spectral_rolloff(waveform),
            "zero_crossing_rate": self.zero_crossing_rate(waveform),
            "lpcs": self.lpcs(waveform),
            "tempogram": self.tempogram(waveform),
            "chromagram": self.chromagram(waveform),
            "pseudo_const_Q_transform": self.pseudo_const_Q_transform(waveform),
            "iirt": self.iirt(waveform),
            "variable_Q_transform": self.variable_Q_transform(waveform),
            "const_Q_chromagram": self.const_Q_chromagram(waveform),
        }

        features.update(self.spectrogram_mel_spectrogram(waveform))

        return features