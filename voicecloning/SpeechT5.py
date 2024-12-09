import logging
import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from scipy.io.wavfile import write
from typing import Tuple
import numpy as np
import argparse
import pandas as pd
from transformers import set_seed

# Configure logging to display detailed output
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)


"""
====================================================
A class for voice cloning using the SpeechT5 model.
====================================================
"""
class VoiceCloner:
    """
    ================================================================================
    Initializes the VoiceCloner, loads the dataset, and sets up the SpeechT5 model.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        split_name (str): Split name of the dataset (e.g., 'test.clean').
        cache_dir (str): Directory to cache the dataset.
        audio_column (str): Name of the column containing audio data.
        text_column (str): Name of the column containing text data.
        speaker_column (str): Name of the column with speaker identifiers.
        id_column (str): Name of the column with sample identifiers.
    ================================================================================
    """
    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        cache_dir: str,
        audio_column: str,
        text_column: str,
        speaker_column: str,
        id_column: str,
    ) -> None:
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.audio_column = audio_column
        self.text_column = text_column
        self.speaker_column = speaker_column
        self.id_column = id_column
        self.dataset = self._load_dataset(dataset_name, split_name, cache_dir)
        self.prompts = pd.read_csv(f'../data/text_prompts/ls-test-clean.csv')
        
        logger.info(f'Initializing SpeechT5 model')
        self._init_speecht5()
        self._clone = self._clone_speecht5


    """
    ===================================================================
    Loads a dataset from Hugging Face and removes unnecessary columns.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        split_name (str): Split name of the dataset.
        cache_dir (str): Path to the cache directory.

    Returns:
        Dataset: A dataset with only the relevant columns.
    ===================================================================
    """
    def _load_dataset(self, dataset_name: str, split_name: str, cache_dir: str) -> Dataset:
        logger.info(f'Loading {split_name} split of {dataset_name} dataset')
        dataset = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir, trust_remote_code=True)
        
        # Check for necessary columns
        required_columns = [self.audio_column, self.text_column, self.speaker_column, self.id_column]
        for col in required_columns:
            if col not in dataset.column_names:
                raise ValueError(f'Required column {col} not found in dataset.')
        
        # Remove unnecessary columns
        to_remove = [c for c in dataset.column_names if c not in required_columns]
        dataset = dataset.remove_columns(to_remove)
        logger.info(f'Removed columns: {to_remove}')
        return dataset

    """
    =============================================================================================
    Initializes components of the SpeechT5 model: classifier, processor, TTS model, and vocoder.
    =============================================================================================
    """
    def _init_speecht5(self) -> None:
        from speechbrain.pretrained.interfaces import EncoderClassifier
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        
        torch.backends.cudnn.deterministic = True
        set_seed(42)
        
        # Speaker classifier for embedding extraction
        self.classifier = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-xvect-voxceleb',
            run_opts={"device": self.device},
            savedir=os.path.join('/tmp', 'speechbrain/spkrec-xvect-voxceleb')
        )
        # Processor for handling inputs to the model
        self.processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        # Text-to-speech model
        self.model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts').to(self.device)
        # HiFi-GAN vocoder for audio post-processing
        self.vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(self.device)


    """
    =====================================================================
    Clones a voice based on a text prompt and audio input.

    Args:
        audio (torch.Tensor): Tensor containing audio data.
        text_prompt (str): Text to be synthesized in the cloned voice.

    Returns:
        Tuple[np.ndarray, int]: Cloned audio data and its sampling rate.
    =====================================================================
    """
    def _clone_speecht5(self, audio: torch.Tensor, text_prompt: str) -> Tuple[np.ndarray, int]:
        torch.backends.cudnn.deterministic = True
        set_seed(42)
        
        # Extract speaker embeddings
        speaker_embeddings = self.classifier.encode_batch(audio)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings[0].view(1, -1)
        
        # Process the text prompt
        inputs = self.processor(text=text_prompt, return_tensors='pt').to(self.device)
        
        # Generate synthetic audio
        cloned_audio = self.model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=self.vocoder)
        return (cloned_audio.view(-1).cpu().numpy(), 16_000)


    """
    ============================================================================
    Generates cloned audio samples based on the dataset and saves them to disk.
    ============================================================================
    """
    def generate(self):
        for sample in tqdm(self.dataset):
            # Prepare file name for saving the cloned audio
            filename = f'{sample[self.speaker_column]}_{sample[self.id_column]}.wav'
            
            # Get the text prompt associated with the current sample
            text_prompt = self.prompts.loc[
                self.prompts['id_sample_to_clone'] == sample[self.id_column],
                'text'
            ].values[0]
            
            # Extract the audio data to be cloned and move it to the appropriate device
            audio_to_clone = torch.tensor(sample[self.audio_column]['array']).to(self.device)
            
            # Get the ID of the sample to compare against for reference audio
            id_to_compare = self.prompts.loc[
                self.prompts['id_sample_to_clone'] == sample[self.id_column],
                'id_sample_to_compare'
            ].values[0]

            # Retrieve the reference sample from the dataset based on the comparison ID
            sample_to_compare = self.dataset.filter(
                lambda example: example[self.id_column] == id_to_compare
            )[0]
            
            # Extract the reference audio and its sampling rate
            audio_to_compare = sample_to_compare[self.audio_column]['array']
            sampling_rate_to_compare = sample_to_compare[self.audio_column]['sampling_rate']
            
            # Save the reference audio to the 'original_samples' directory
            write(f'original_samples/{filename}', sampling_rate_to_compare, audio_to_compare)

            # Clone the audio using the text prompt and save it to the 'cloned_samples' directory
            cloned_audio, cloned_sampling_rate = self._clone(audio_to_clone, text_prompt)
            write(f'cloned_samples/{filename}', cloned_sampling_rate, cloned_audio)
            

"""
================================================================
Parses input arguments for the script.

Returns:
    argparse.Namespace: Parsed arguments as a namespace object.
================================================================
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='openslr/librispeech_asr')
    parser.add_argument('--split_name', type=str, default='test.clean')
    parser.add_argument('--cache_dir', type=str, default='/projects/csi_huggingface/cache')
    parser.add_argument('--audio_column', type=str, default='audio')
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--speaker_column', type=str, default='speaker_id')
    parser.add_argument('--id_column', type=str, default='id')
    
    return parser.parse_args()


"""
=================================================================================================
Main function of the script: initializes the VoiceCloner class and triggers the cloning process.

Args:
    args (argparse.Namespace): Input arguments.
=================================================================================================
"""
def main(args: argparse.Namespace):
    if not os.path.isdir('original_samples'):
        os.makedirs('original_samples', exist_ok=True)
        logger.info('Created directory: original_samples')
    
    if not os.path.isdir('cloned_samples'):
        os.makedirs('cloned_samples', exist_ok=True)
        logger.info('Created directory: cloned_samples')

    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f'Created directory: {args.cache_dir}')

    logger.info(f'Voice cloning arguments: [{args}]')
    voice_cloner = VoiceCloner(
        args.dataset_name,
        args.split_name,
        args.cache_dir,
        args.audio_column,
        args.text_column,
        args.speaker_column,
        args.id_column,
    )
    voice_cloner.generate()

if __name__ == '__main__':
    args = parse_args()
    main(args)