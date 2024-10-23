import numpy as np
from typing import Union
from dataclasses import dataclass
import scipy.stats
import torch
import librosa
import scipy


@dataclass
class LibrosaWrapperOutput:
    """
    A data class to store the outputs of various audio processing transformations.

    Attributes:
        pitch (np.ndarray): Extracted pitch from the input waveform.
        spectrogram (np.ndarray): Spectrogram of the input waveform.
        spectrogram_delta1 (np.ndarray): First-order delta of the spectrogram.
        spectrogram_delta2 (np.ndarray): Second-order delta of the spectrogram.
        mel_spectrogram (np.ndarray): Mel spectrogram of the input waveform.
        mel_spectrogram_delta1 (np.ndarray): First-order delta of the mel spectrogram.
        mel_spectrogram_delta2 (np.ndarray): Second-order delta of the mel spectrogram.
        mfcc (np.ndarray): Mel-frequency cepstral coefficients.
        mfcc_delta1 (np.ndarray): First-order delta of the MFCC.
        mfcc_delta2 (np.ndarray): Second-order delta of the MFCC.
        rms (np.ndarray): Root mean square (RMS) energy.
        spectral_centroid (np.ndarray): Spectral centroid of the input waveform.
        spectral_bandwidth (np.ndarray): Spectral bandwidth of the input waveform.
        spectral_contrast (np.ndarray): Spectral contrast of the input waveform.
        spectral_flatness (np.ndarray): Spectral flatness of the input waveform.
        spectral_rolloff (np.ndarray): Spectral rolloff of the input waveform.
        zero_crossing_rate (np.ndarray): Zero crossing rate of the input waveform.
        static_tempo (np.ndarray): Static tempo estimation.
        static_tempo_uniform (np.ndarray): Static tempo estimation with uniform prior.
        dynamic_tempo (np.ndarray): Dynamic tempo estimation.
        dynamic_tempo_lognorm (np.ndarray): Dynamic tempo estimation with log-normal prior.
        linear_prediction_coefficients (np.ndarray): Linear prediction coefficients.
        tempogram (np.ndarray): Tempogram of the input waveform.
        chromagram (np.ndarray): Chromagram of the input waveform.
        pseudo_const_Q_transform (np.ndarray): Pseudo constant-Q transform.
        fast_mellin_transform (np.ndarray): Fast Mellin transform.
        iirt (np.ndarray): Inverse iterative real-time transform.
        variable_Q_transform (np.ndarray): Variable-Q transform.
        autocorrelation (np.ndarray): Autocorrelation of the input waveform.
        clicks (np.ndarray): Clicks detected in the input waveform.
        mag_specgram_griffin_lim (np.ndarray): Magnitude spectrogram reconstructed using Griffin-Lim algorithm.
        const_Q_chromagram (np.ndarray): Constant-Q chromagram.
    """
    pitch: np.ndarray = None
    spectrogram: np.ndarray = None
    spectrogram_delta1: np.ndarray = None
    spectrogram_delta2: np.ndarray = None
    mel_spectrogram: np.ndarray = None
    mel_spectrogram_delta1: np.ndarray = None
    mel_spectrogram_delta2: np.ndarray = None
    mfcc: np.ndarray = None
    mfcc_delta1: np.ndarray = None
    mfcc_delta2: np.ndarray = None
    rms: np.ndarray = None
    spectral_centroid: np.ndarray = None
    spectral_bandwidth: np.ndarray = None
    spectral_contrast: np.ndarray = None
    spectral_flatness: np.ndarray = None
    spectral_rollof: np.ndarray = None
    zero_crossing_rate: np.ndarray = None
    static_tempo: np.ndarray = None
    static_tempo_uniform: np.ndarray = None
    dynamic_tempo: np.ndarray = None
    dynamic_tempo_lognorm: np.ndarray = None
    linear_prediction_coefficients: np.ndarray = None
    tempogram: np.ndarray = None
    chromagram: np.ndarray = None
    pseudo_const_Q_transform: np.ndarray = None
    # fast_mellin_transform: np.ndarray = None
    iirt: np.ndarray = None
    variable_Q_transform: np.ndarray = None
    # autocorrelation: np.ndarray = None
    clicks: np.ndarray = None
    mag_specgram_griffin_lim: np.ndarray = None
    # const_Q_mag_specgram_griffin_lim: np.ndarray = None
    const_Q_chromagram: np.ndarray = None


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

    def __call__(self, waveform: Union[torch.Tensor, np.ndarray, list], sampling_rate: int, return_dict: bool = False) -> LibrosaWrapperOutput:
        """
        Process the input waveform through various Librosa transformations.

        Args:
            waveform (Union[torch.Tensor, np.ndarray, list]): The input audio waveform.
            sampling_rate (int): The sampling rate of the input audio.

        Returns:
            LibrosaWrapperOutput: The processed audio features.
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

        if not return_dict:
            return LibrosaWrapperOutput(
                pitch=librosa.pyin(y=waveform, sr=self.sampling_rate, fmin=65, fmax=2093)[0],
                spectrogram=specgram,
                spectrogram_delta1=specgram_delta1,
                spectrogram_delta2=librosa.feature.delta(specgram_delta1),
                mel_spectrogram=librosa.power_to_db(mel_specgram, ref=np.max),
                mel_spectrogram_delta1=mel_specgram_delta1,
                # mel_spectrogram_delta2=librosa.feature.delta(mel_specgram_delta1),
                mfcc=mfcc,
                mfcc_delta1=mfcc_delta1,
                mfcc_delta2=librosa.feature.delta(mfcc_delta1),
                rms=librosa.feature.rms(y=waveform),
                spectral_centroid=librosa.feature.spectral_centroid(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
                spectral_bandwidth=librosa.feature.spectral_bandwidth(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
                spectral_contrast=librosa.feature.spectral_contrast(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
                spectral_flatness=librosa.feature.spectral_flatness(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length),
                spectral_rollof=librosa.feature.spectral_rolloff(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
                zero_crossing_rate=librosa.feature.zero_crossing_rate(y=waveform, frame_length=self.n_fft, hop_length=self.hop_length),
                static_tempo=librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate),
                static_tempo_uniform=librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, prior=scipy.stats.uniform(30, 300)),
                dynamic_tempo=librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, aggregate=None),
                dynamic_tempo_lognorm=librosa.feature.tempo(onset_envelope=onset_env, sr=self.sampling_rate, aggregate=None, prior=scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)),
                linear_prediction_coefficients=librosa.lpc(y=waveform, order=2),
                tempogram=librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sampling_rate, hop_length=self.hop_length),
                chromagram=librosa.feature.chroma_stft(y=waveform, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length),
                pseudo_const_Q_transform=np.abs(librosa.pseudo_cqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length, fmin=65, n_bins=60 * 2, bins_per_octave=12 * 2)),
                # fast_mellin_transform=np.abs(librosa.fmt(librosa.util.normalize(fmt_ac), n_fmt=512)),
                iirt=librosa.amplitude_to_db(np.abs(librosa.iirt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
                variable_Q_transform=librosa.amplitude_to_db(np.abs(librosa.vqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
                # autocorrelation=librosa.autocorrelate(y=waveform),
                # clicks=librosa.clicks(frames=librosa.beat.beat_track(y=waveform, sr=self.sampling_rate)[1], sr=self.sampling_rate),
                # mag_specgram_griffin_lim=librosa.griffinlim(specgram),
                # const_Q_mag_specgram_griffin_lim=librosa.griffinlim_cqt(np.abs(librosa.cqt(y=waveform, sr=self.sampling_rate, bins_per_octave=36, n_bins=36*7)), sr=self.sampling_rate, bins_per_octave=36),
                const_Q_chromagram=librosa.feature.chroma_cqt(y=waveform, sr=self.sampling_rate),
            )
        else:
            return {
                'pitch': librosa.pyin(y=waveform, sr=self.sampling_rate, fmin=65, fmax=2093)[0],
                'spectrogram': specgram,
                'spectrogram_delta1': specgram_delta1,
                # 'spectrogram_delta2': librosa.feature.delta(specgram_delta1),
                'mel_spectrogram': librosa.power_to_db(mel_specgram, ref=np.max),
                'mel_spectrogram_delta1': mel_specgram_delta1,
                'mel_spectrogram_delta2': librosa.feature.delta(mel_specgram_delta1),
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
                # 'fast_mellin_transform': np.abs(librosa.fmt(librosa.util.normalize(fmt_ac), n_fmt=512)),
                'iirt': librosa.amplitude_to_db(np.abs(librosa.iirt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
                'variable_Q_transform': librosa.amplitude_to_db(np.abs(librosa.vqt(y=waveform, sr=self.sampling_rate, hop_length=self.hop_length)), ref=np.max),
                # 'autocorrelation': librosa.autocorrelate(y=waveform),
                # 'clicks': librosa.clicks(frames=librosa.beat.beat_track(y=waveform, sr=self.sampling_rate)[1], sr=self.sampling_rate),
                # 'mag_specgram_griffin_lim': librosa.griffinlim(specgram),
                # 'const_Q_mag_specgram_griffin_lim'=librosa.griffinlim_cqt(np.abs(librosa.cqt(y=waveform, sr=self.sampling_rate, bins_per_octave=36, n_bins=36*7)), sr=self.sampling_rate, bins_per_octave=36),
                'const_Q_chromagram': librosa.feature.chroma_cqt(y=waveform, sr=self.sampling_rate),
            }