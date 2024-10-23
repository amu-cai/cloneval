""" Feature extractor class for Emotion Transfer Benchmark (ETB) """

from typing import Union, List, Optional

import numpy as np 
import torch 
import torchaudio
import whisper

import logging


logger = logging.getLogger(__name__)


class ETBFeatureExtractor():
    def __init__(
        self,
        whisper_model_name: str = 'tiny',
        sampling_rate: int = 16_000,
        n_mfcc: int = 12,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
        center: bool = False,
        win_length: int = 5,
        device: torch.device = None
    ):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.sampling_rate = sampling_rate
        self.target_length = sampling_rate * chunk_length
        self.whisper_model = self.get_whisper_model(whisper_model_name).to(self.device)
        self.mfcc = self.get_mfcc(sampling_rate, n_mfcc, n_fft, hop_length, center).to(self.device)
        self.compute_deltas = self.get_deltas(win_length).to(self.device)

    def get_whisper_model(self, whisper_model_name):
        whisper_model = whisper.load_model(whisper_model_name)
        for param in whisper_model.parameters():
            param.requires_grad = False
        return whisper_model
    
    def get_mfcc(self, sampling_rate: int, n_mfcc: int, n_fft: int, hop_length: int, center: bool):
        return torchaudio.transforms.MFCC(
            sample_rate=sampling_rate, 
            n_mfcc=n_mfcc, 
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "center": center},
        )
    
    def get_deltas(self, win_length: int):
        return torchaudio.transforms.ComputeDeltas(win_length=win_length)
    
    def process_batch(self, batch: List[dict]) -> dict:
        batch_inputs = [[sample['audio']['array'], sample['audio']['sampling_rate']] for sample in batch]
        batch_labels = [sample['emotion'] for sample in batch]
        
        input_features = []

        for audio, sampling_rate in batch_inputs:
            features = self.__call__(audio, sampling_rate)
            input_features.append(features)

        input_features = torch.cat(input_features, dim=0)

        return {
            'input_features': input_features,
            'labels': batch_labels,
        }
    
    def __call__(
        self, 
        audio: Union[torch.Tensor, np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
    ) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32).view(1, -1)

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                logger.warning(f"The required sampling rate is {self.sampling_rate}. Resampling audio from {sampling_rate} Hz to {self.sampling_rate} Hz.")
                audio = torchaudio.functional.resample(waveform=audio, orig_freq=sampling_rate, new_freq=self.sampling_rate)

        if audio.size(-1) < self.target_length:
            audio = [audio] * (self.target_length // audio.size(-1) + 1)
            audio = torch.cat(audio, dim=-1)
            audio = audio[:, : self.target_length]
        elif audio.size(-1) > self.target_length:
            audio = audio[:, : self.target_length]

        audio = audio.to(self.device)

        mfcc = self.mfcc(audio)
        delta1 = self.compute_deltas(mfcc)
        delta2 = self.compute_deltas(delta1)
        spectral_features = torch.cat((mfcc, delta1, delta2), dim=-2)

        mel = whisper.log_mel_spectrogram(audio)
        whisper_values = self.whisper_model.encoder(mel).transpose(1, -1)
        whisper_features = torch.cat((whisper_values, whisper_values), dim=-1)

        input_features = torch.cat((spectral_features, whisper_features[:, :, :-2]), dim=-2)
        
        return input_features.unsqueeze(0).transpose(0, 1)
