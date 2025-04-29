import os
import torch
import numpy as np
import librosa
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import WavLMForXVector, AutoFeatureExtractor

from .librosa_wrapper import LibrosaWrapper


SAMPLING_RATE = 16_000

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class ClonEval:
    def __init__(self) -> None:
        self.librosa_wrapper = LibrosaWrapper(sampling_rate=SAMPLING_RATE)
        self.feature_list = None
        self._init_wavlm()

    def _init_wavlm(self) -> None:
        """Load pretrained WavLM model for speaker embedding extraction."""
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    def _read_audio(self, path: str) -> np.ndarray:
        """Read and resample audio to target sampling rate."""
        waveform, _ = librosa.load(path, sr=SAMPLING_RATE)
        return waveform
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature array, handling NaNs and reshaping."""
        features = np.nan_to_num(features)
        return features.reshape(1, -1) if features.ndim < 2 else features
    
    def _calc_cossim(self, o: np.ndarray, c: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        o, c = self._normalize_features(o), self._normalize_features(c)
        return np.mean(cosine_similarity(o, c)).item()
    
    def _get_speaker_embeds(self, x: np.ndarray) -> torch.Tensor:
        """Extract speaker embedding from a waveform using WavLM."""
        inputs = self.feature_extractor(x, sampling_rate=SAMPLING_RATE, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.wavlm(**inputs)        
        return outputs.embeddings.squeeze().cpu().numpy()

    def eval_sample(self, original_path: str, cloned_path: str):
        """
        Evaluate similarity between an original and cloned audio sample.
        Returns a dictionary of cosine similarities per feature.
        """
        original_waveform = self._read_audio(original_path)
        cloned_waveform = self._read_audio(cloned_path)

        min_len = min(len(original_waveform), len(cloned_waveform))
        original_waveform = original_waveform[:min_len]
        cloned_waveform = cloned_waveform[:min_len]

        if np.abs(librosa.stft(original_waveform)).shape[1] < 9:
            if self.feature_list is None:
                return {}
            return {f: 0.0 for f in self.feature_list + ["wavlm"]}

        original_features = self.librosa_wrapper(waveform=original_waveform, sampling_rate=SAMPLING_RATE)
        cloned_features = self.librosa_wrapper(waveform=cloned_waveform, sampling_rate=SAMPLING_RATE)

        if self.feature_list is None:
            self.feature_list = list(original_features.keys())

        results = {
            feature: self._calc_cossim(original_features[feature], cloned_features[feature])
            for feature in self.feature_list
        }

        results["wavlm"] = self._calc_cossim(
            self._get_speaker_embeds(original_waveform),
            self._get_speaker_embeds(cloned_waveform),
        )

        return results

    def evaluate(
        self, 
        original_dir: str, 
        cloned_dir: str, 
        evaluate_emotion_transfer: bool = False, 
        output_dir: str = ".",
    ) -> None:
        """
        Evaluate all audio files in `original_dir` and `cloned_dir`, comparing original and cloned
        samples. Saves full and aggregated results as CSV files.
        """
        filenames = sorted(os.listdir(original_dir))
        all_results = []
        for filename in tqdm(filenames, desc="Evaluating"):
            original_path = os.path.join(original_dir, filename)
            cloned_path = os.path.join(cloned_dir, filename)

            result = {"filename": filename}
            features = self.eval_sample(original_path, cloned_path)
            if not features:
                continue

            result.update(features)

            if evaluate_emotion_transfer:
                result["emotion"] = filename.replace(".wav", "").split("_")[-1]
            
            all_results.append(result)
        
        results_ds = Dataset.from_list(all_results)
        results_ds.to_csv("results.csv")

        if evaluate_emotion_transfer:
            emotions = set(results_ds["emotion"])
            aggregated = []

            for emotion in emotions:
                filtered = results_ds.filter(lambda x: x["emotion"] == emotion)
                avg = {
                    k: np.mean(filtered[k]) if k not in {"filename", "emotion"} else emotion
                    for k in results_ds.column_names if k != "filename"
                }
                aggregated.append(avg)
            
            avg_all = {
                k: np.mean(results_ds[k]) if k not in {"filename", "emotion"} else "all"
                for k in results_ds.column_names if k != "filename"
            }
            aggregated.append(avg_all)

            Dataset.from_list(aggregated).to_csv(f"{output_dir}/aggregated_results.csv")
        else:
            avg = {
                k: [np.mean(results_ds[k])] for k in results_ds.column_names if k != "filename"
            }
            avg["emotion"] = ["all"]
            Dataset.from_dict(avg).to_csv(f"{output_dir}/aggregated_results.csv")