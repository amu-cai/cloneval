import logging
import os
import sys

import torch

from dataclasses import dataclass, field
from transformers import HfArgumentParser

from datasets import load_dataset, Dataset

from tqdm import tqdm

from scipy.io.wavfile import write

from typing import Tuple
import numpy as np

import argparse


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)


class VoiceCloner:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        split_name: str,
        output_dir: str,
        cache_dir: str,
        audio_column: str,
        text_column: str,
        speaker_column: str,
    ) -> None:
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = output_dir
        self.audio_column = audio_column
        self.text_column = text_column
        self.speaker_column = speaker_column
        self.dataset = self._load_dataset(dataset_name, split_name, cache_dir)
        
        logger.info(f'Initializing {model_name} model')
        self.model_name = model_name
        if self.model_name == 'SpeechT5':
            self._init_speecht5()
            self._clone = self._clone_speecht5
        elif self.model_name == 'VALL-E_X':
            self._init_vallex()
            self._clone = self._clone_vallex
        elif self.model_name == 'WhisperSpeech':
            self._init_whisperspeech()
            self._clone = self._clone_whisperspeech

    def _load_dataset(self, dataset_name: str, split_name: str, cache_dir: str) -> Dataset:
        logger.info(f'Loading {split_name} split of {dataset_name} dataset')
        dataset = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
        if self.audio_column not in dataset.column_names:
            raise ValueError(f'Audio column {self.audio_column} not available in {dataset_name} dataset.')
        if self.text_column not in dataset.column_names:
            raise ValueError(f'Text column {self.text_column} not available in {dataset_name} dataset.')
        if self.speaker_column not in dataset.column_names:
            raise ValueError(f'Speaker column {self.speaker_column} not available in {dataset_name} dataset.')
        to_remove = [c for c in dataset.column_names if c not in [self.audio_column, self.text_column, self.speaker_column]]
        # to_remove = [c for c in dataset.column_names if c not in [self.audio_column, self.text_column, self.speaker_column, 'emotion']]
        dataset = dataset.remove_columns(to_remove)
        logger.info(f'Removing columns: {to_remove}')
        return dataset

    def _init_speecht5(self) -> None:
        from speechbrain.pretrained.interfaces import EncoderClassifier
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        self.classifier = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-xvect-voxceleb', 
            run_opts={"device": self.device}, 
            savedir=os.path.join('/tmp', 'speechbrain/spkrec-xvect-voxceleb'),
        )
        self.processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        self.model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(self.device)

    def _init_vallex(self) -> None:
        sys.path.insert(0, '/projects/etb/etb/VALL-E-X')
        os.chdir('/projects/etb/etb/VALL-E-X')
        from utils.prompt_making import make_prompt
        from utils.generation import SAMPLE_RATE, generate_audio, preload_models
        preload_models()
        sys.path.insert(0, '/projects/etb/etb/scripts')
        os.chdir('/projects/etb/etb/scripts')
    
    def _init_whisperspeech(self) -> None:
        from speechbrain.pretrained.interfaces import EncoderClassifier
        from whisperspeech.pipeline import Pipeline

        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
        if self.pipe.encoder is None:
            self.device = self.pipe.device
            if self.device == 'mps': self.device = 'cpu'
            self.pipe.encoder = EncoderClassifier.from_hparams(
                'speechbrain/spkrec-ecapa-voxceleb', 
                savedir=os.path.expanduser("~/.cache/speechbrain/"), 
                run_opts={"device": self.device},
            )

    def _clone_speecht5(self, audio: torch.Tensor, sampling_rate: int, text_prompt: str) -> Tuple[np.ndarray, int]:
        speaker_embeddings = self.classifier.encode_batch(audio)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        if speaker_embeddings.size(0) > 1:
            speaker_embeddings = speaker_embeddings[0]
        speaker_embeddings = speaker_embeddings.view(1, -1)
        inputs = self.processor(text=text_prompt, return_tensors='pt').to(self.device)
        cloned_audio = self.model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=self.vocoder)
        return (cloned_audio.view(-1).cpu().numpy(), 16_000)

    def _clone_vallex(self, audio: str, sampling_rate: int, text_prompt: str) -> Tuple[np.ndarray, int]:
        sys.path.insert(0, '/projects/etb/etb/VALL-E-X')
        os.chdir('/projects/etb/etb/VALL-E-X')
        from utils.prompt_making import make_prompt
        from utils.generation import generate_audio
        make_prompt(name="promt",
                    audio_prompt_path=f'../scripts/{audio}',
                    transcript=text_prompt)
        
        cloned_audio = generate_audio(text_prompt, prompt='promt')
        sys.path.insert(0, '/projects/etb/etb/scripts')
        os.chdir('/projects/etb/etb/scripts')

        return (cloned_audio, 16_000)

    def _clone_whisperspeech(
        self, 
        audio: torch.Tensor,
        sampling_rate: int,
        text_prompt: str,
        cps: int = 15, 
        lang: str = 'pln',
    ) -> Tuple[np.ndarray, int]:
        n_frames = audio.size(-1) * 30
        audio = audio[:n_frames]
        audio = self.pipe.encoder.audio_normalizer(audio, sampling_rate)
        speaker_embeddings = self.pipe.encoder.encode_batch(audio.unsqueeze(0))[0, 0]
        stoks = self.pipe.t2s.generate(text_prompt, cps=cps, lang=lang, step=None)[0]
        stoks = stoks[stoks != 512]
        atoks = self.pipe.s2a.generate(stoks, speaker_embeddings.unsqueeze(0), step=None)
        cloned_audio = self.pipe.vocoder.decode(atoks)
        return (cloned_audio.view(-1).cpu().numpy(), 24_000)

    def generate(self):
        speakers = {}
        for sample in tqdm(self.dataset):
            speaker_id = sample[self.speaker_column]
            if speakers.get(speaker_id) is None:
                speakers[speaker_id] = 0
            # filename = f'{speaker_id}_{speakers[speaker_id]}_{sample['emotion']}.wav'
            filename = f'{speaker_id}_{speakers[speaker_id]}.wav'
            speakers[speaker_id] += 1
            audio = sample[self.audio_column]['array']
            sampling_rate = sample[self.audio_column]['sampling_rate']
            write(f'{self.output_dir}/original_samples/{filename}', sampling_rate, audio)
            text_prompt = sample[self.text_column].strip()
            if self.model_name=='VALL-E_X':
                audio = f'{self.output_dir}/original_samples/{filename}'
            else:
                audio = torch.tensor(sample[self.audio_column]['array']).to(self.device)
            try:    
                cloned_audio, cloned_sampling_rate = self._clone(audio, sampling_rate, text_prompt)
                write(f'{self.output_dir}/cloned_samples/{filename}', cloned_sampling_rate, cloned_audio)
            except Exception as e:
                sys.path.insert(0, '/projects/etb/etb/scripts')
                os.chdir('/projects/etb/etb/scripts')
                print(f'Cloning error {e}')
            


def parse_args():
    parser = argparse.ArgumentParser()

    group_model = parser.add_argument_group(
        'Model', ''
    )
    group_model.add_argument(
        '--model_name',
        type=str,
        default=None,
        choices=['SpeechT5', 'VALL-E_X', 'WhisperSpeech'],
        help='',
    )
    group_model.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help=''
    )

    group_dataset = parser.add_argument_group(
        'Dataset', ''
    )
    group_dataset.add_argument(
        '--dataset_name',
        type=str,
        default='openslr/librispeech_asr',
        help='Defaults to `openslr/librispeech_asr`.',
    )
    group_dataset.add_argument(
        '--split_name',
        type=str,
        default='test.clean',
        help='Defaults to `test.clean`.',
    )
    group_dataset.add_argument(
        '--output_dir',
        type=str,
        default='None',
        help='',
    )
    group_dataset.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache',     # /projects/csi_huggingface/cache
        help='Defaults to `~/.cache`.',
    )
    group_dataset.add_argument(
        '--audio_column',
        type=str,
        default='audio',
        help='Defaults to `audio`.',
    )
    group_dataset.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='Defaults to `text`.',
    )
    group_dataset.add_argument(
        '--speaker_column',
        type=str,
        default='speaker_id',
        help='Defaults to `speaker_id`.',
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    # Check if output_dir exists, if not, create directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f'Created directory: {args.output_dir}')
    
    if not os.path.isdir(f'{args.output_dir}/original_samples'):
        os.makedirs(f'{args.output_dir}/original_samples', exist_ok=True)
        logger.info(f'Created directory: {args.output_dir}/original_samples')
    
    if not os.path.isdir(f'{args.output_dir}/cloned_samples'):
        os.makedirs(f'{args.output_dir}/cloned_samples', exist_ok=True)
        logger.info(f'Created directory: {args.output_dir}/cloned_samples')

    # If cache_dir is not None, check if cache_dir exists, if not, create directory
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f'Created directory: {args.cache_dir}')

    logger.info(f'Voice cloning arguments: [{args}]')

    voice_cloner = VoiceCloner(
        args.model_name,
        args.dataset_name,
        args.split_name,
        args.output_dir,
        args.cache_dir,
        args.audio_column,
        args.text_column,
        args.speaker_column,
    )
    voice_cloner.generate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
