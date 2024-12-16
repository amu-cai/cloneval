import logging
import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from scipy.io.wavfile import write
from typing import Tuple
import numpy as np
import pandas as pd
from transformers import set_seed

# Configure logging to display detailed output
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s | %(asctime)s | %(message)s', level=logging.INFO)


class VoiceCloner:
    """
    A class for voice cloning using the SpeechT5 model.
    """
    def __init__(self) -> None:
        """
        Initializes the VoiceCloner, loads the dataset, and sets up the SpeechT5 model.
        """
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.audio_column = 'audio'
        self.text_column = 'text'
        self.speaker_column = 'speaker_id'
        self.id_column = 'id'
        self.dataset = load_dataset('openslr/librispeech_asr', split='test.clean', streaming=True, trust_remote_code=True)
        self.prompts = pd.read_csv(f'../data/text_prompts/ls-test-clean.csv')
        
        logger.info(f'Initializing SpeechT5 model')
        self._init_speecht5()
        self._clone = self._clone_speecht5


    def _init_speecht5(self) -> None:
        """
        Initializes components of the SpeechT5 model: classifier, processor, TTS model, and vocoder.
        """
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


    def _clone_speecht5(self, audio: torch.Tensor, text_prompt: str) -> Tuple[np.ndarray, int]:
        """
        Clones a voice based on a text prompt and audio input.

        Args:
            audio (torch.Tensor): Tensor containing audio data.
            text_prompt (str): Text to be synthesized in the cloned voice.

        Returns:
            Tuple[np.ndarray, int]: Cloned audio data and its sampling rate.
        """
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


    def generate(self):
        """
        Generates cloned audio samples based on the dataset and saves them to disk.
        """
        # Tracks the current loop iteration
        current_iteration = 0

        for sample in tqdm(self.dataset):
            # Prepare file name for saving the cloned audio
            filename = f'{sample[self.speaker_column]}_{sample[self.id_column]}.wav'
            
            # Get the text prompt associated with the current sample
            model_input_text_prompt = self.prompts.loc[
                self.prompts['id_sample_to_clone'] == sample[self.id_column],
                'text'
            ].values[0]
            
            # Extract the audio data to be cloned and move it to the appropriate device
            model_input_audio = torch.tensor(sample[self.audio_column]['array']).to(self.device)
            
            # Get the ID of the sample to compare against for reference audio
            id_sample_to_compare = self.prompts.loc[
                self.prompts['id_sample_to_clone'] == sample[self.id_column],
                'id_sample_to_compare'
            ].values[0]

            # Retrieve the reference sample from the dataset based on the comparison ID
            for proposed_sample_to_compare in self.dataset:
                if proposed_sample_to_compare[self.id_column] == id_sample_to_compare:
                    sample_to_compare = proposed_sample_to_compare

            # sample_to_compare = self.dataset.filter(
            #     lambda example: example[self.id_column] == id_sample_to_compare
            # )[0]
            
            # Extract the reference audio and its sampling rate
            audio_to_compare = sample_to_compare[self.audio_column]['array']
            sampling_rate_to_compare = sample_to_compare[self.audio_column]['sampling_rate']
            
            # Save the reference audio to the 'original_samples' directory
            write(f'original_samples/{filename}', sampling_rate_to_compare, audio_to_compare)

            # Clone the audio using the text prompt and save it to the 'cloned_samples' directory
            cloned_audio, cloned_sampling_rate = self._clone(model_input_audio, model_input_text_prompt)
            write(f'cloned_samples/{filename}', cloned_sampling_rate, cloned_audio)

            # Ends the loop when iteration reaches 5
            current_iteration += 1
            if current_iteration == 5:
                break
            

def main():
    """
    Main function of the script: initializes the VoiceCloner class and triggers the cloning process.
    """
    if not os.path.isdir('original_samples'):
        os.makedirs('original_samples', exist_ok=True)
        logger.info('Created directory: original_samples')
    
    if not os.path.isdir('cloned_samples'):
        os.makedirs('cloned_samples', exist_ok=True)
        logger.info('Created directory: cloned_samples')

    voice_cloner = VoiceCloner()
    voice_cloner.generate()

if __name__ == '__main__':
    main()