import numpy as np
import torch
import librosa
from typing import Union, Dict


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
        self.n_fft = n_fft
        self.hop_length = hop_length

    def preprocess(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int) -> np.ndarray:
        """Convert waveform to NumPy array and resample if necessary."""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        elif isinstance(waveform, list):
            waveform = np.array(waveform)
        
        if sampling_rate != self.sampling_rate:
            waveform = librosa.resample(y=waveform, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        
        return waveform

    def __call__(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int) -> dict:
        """Extract audio features from the input waveform."""
        waveform = self.preprocess(waveform, sampling_rate)

        stft = librosa.stft(y=waveform, hop_length=self.hop_length, n_fft=self.n_fft)
        spectrogram = np.abs(stft)

        return {
            "pitch": librosa.pyin(y=waveform, sr=self.sampling_rate, fmin=65, fmax=2093)[0],
            "spectrogram": spectrogram,
            "mel_spectrogram": librosa.power_to_db(librosa.feature.melspectrogram(S=spectrogram, sr=self.sampling_rate), ref=np.max),
            "mfccs": librosa.feature.mfcc(y=waveform, sr=self.sampling_rate, n_mfcc=13),
            "rms": librosa.feature.rms(y=waveform),
            "spectral_centroid": librosa.feature.spectral_centroid(S=spectrogram, sr=self.sampling_rate),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=spectrogram, sr=self.sampling_rate),
            "spectral_contrast": librosa.feature.spectral_contrast(S=spectrogram, sr=self.sampling_rate),
            "spectral_flatness": librosa.feature.spectral_flatness(S=spectrogram),
            "spectral_rolloff": librosa.feature.spectral_rolloff(S=spectrogram, sr=self.sampling_rate),
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(y=waveform, frame_length=self.n_fft, hop_length=self.hop_length),
            "lpcs": librosa.lpc(y=waveform, order=2),
            "tempogram": librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=waveform, sr=self.sampling_rate), sr=self.sampling_rate, hop_length=self.hop_length),
            "chromagram": librosa.feature.chroma_stft(S=spectrogram, sr=self.sampling_rate),
            "const_Q_chromagram": librosa.feature.chroma_cqt(y=waveform, sr=self.sampling_rate),
            "pseudo_const_Q_transform": np.abs(librosa.pseudo_cqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)),
            "iirt": librosa.amplitude_to_db(np.abs(librosa.iirt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
            "variable_Q_transform": librosa.amplitude_to_db(np.abs(librosa.vqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
        }