import torch
import os
from librosa import load
from datasets import Dataset
import numpy as np

from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity

from .librosa_wrapper import LibrosaWrapper
from .wavlm import WavLM, WavLMConfig


SAMPLING_RATE = 16_000

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ClonEval:
    def __init__(self) -> None:
        self.librosa_wrapper = LibrosaWrapper(sampling_rate=SAMPLING_RATE)
        self._init_wavlm()

    def _init_wavlm(self) -> None:
        wavlm_checkpoint = torch.load("./checkpoints/WavLM-Large.pt", weights_only=True)
        self.wavlm_config = WavLMConfig(wavlm_checkpoint["cfg"])
        self.wavlm = WavLM(self.wavlm_config).to(DEVICE)
        self.wavlm.load_state_dict(wavlm_checkpoint["model"])

    def _read_audio(self, pth: str) -> np.ndarray:
        waveform, _ = load(pth, sr=SAMPLING_RATE)
        return waveform
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        features = np.nan_to_num(features)
        if features.ndim < 2:
            features = features.reshape(1, -1)
        return features
    
    def _calc_cossim(self, orig_features: np.ndarray, clon_features: np.ndarray) -> float:
        orig_features, clon_features = self._normalize_features(orig_features), self._normalize_features(clon_features)
        cossim = np.mean(cosine_similarity(orig_features, clon_features)).item()
        return cossim
    
    def _get_speaker_embeds(self, x: np.ndarray) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.float32).view(1, -1).to(DEVICE)
        if self.wavlm_config.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        speaker_embeds, _ = self.wavlm.extract_features(
            x, 
            output_layer=self.wavlm.cfg.encoder_layers, 
            ret_layer_results=True,
        )[0]
        speaker_embeds = speaker_embeds.squeeze(0).detach().cpu().numpy()
        return speaker_embeds

    def eval_sample(self, orig_pth: str, clon_pth: str):
        orig_waveform, clon_waveform = self._read_audio(orig_pth), self._read_audio(clon_pth)

        if len(orig_waveform) < len(clon_waveform):
            clon_waveform = clon_waveform[:len(orig_waveform)]
        elif len(orig_waveform) > len(clon_waveform):
            orig_waveform = orig_waveform[:len(clon_waveform)]

        orig_features = self.librosa_wrapper(waveform=orig_waveform, sampling_rate=SAMPLING_RATE)
        clon_features = self.librosa_wrapper(waveform=clon_waveform, sampling_rate=SAMPLING_RATE)

        results = {}

        for feature in orig_features:
            orig_f = self._normalize_features(orig_features[feature])
            clon_f = self._normalize_features(clon_features[feature])
            cossim = self._calc_cossim(orig_f, clon_f)
            results[feature] = cossim

        results["wavlm"] = self._calc_cossim(
            orig_features=self._get_speaker_embeds(orig_waveform),
            clon_features=self._get_speaker_embeds(clon_waveform),
        )

        return results

    def evaluate(self, orig_dir: str, clon_dir: str, use_emotion: bool = False) -> None:
        filenames = os.listdir(orig_dir)
        results = Dataset.from_dict({"filename": filenames})
        if use_emotion:
            results = results.map(lambda x: {"emotion": x["filename"].replace(".wav", "").split("_")[-1]})
        results = results.map(lambda x: self.eval_sample(orig_pth=f"{orig_dir}/{x['filename']}", clon_pth=f"{clon_dir}/{x['filename']}"))
        if use_emotion:
            emotions = set(results["emotion"])
            aggregated_results = []
            for emotion in emotions:
                emo_results = results.filter(lambda x: x["emotion"] == emotion)
                res = {"emotion": emotion}
                res.update({feature: np.mean(emo_results[feature]).item() for feature in emo_results.column_names if feature not in {"emotion", "filename"}})
                aggregated_results.append(res)
            aggregated_results = Dataset.from_list(aggregated_results)
        else:
            aggregated_results = Dataset.from_dict({
                feature: [np.mean(results[feature]).item()] for feature in results.column_names if feature not in {"emotion", "filename"}
            })
        results.to_csv("./results.csv")
        aggregated_results.to_csv("./aggregated_results.csv")