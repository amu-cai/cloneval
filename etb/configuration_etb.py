""" Emotion Transfer Benchmark (ETB) model configuration """

from transformers.configuration_utils import PretrainedConfig
import json
from typing import Tuple, List


class ETBConfig(PretrainedConfig):
    model_type = 'etb_model'

    def __init__(
        self, 
        emotions: List[str] = [
            'anger', 'boredom', 'calm', 'disgust',
            'fear', 'happiness', 'neutral', 'pleasant-surprise',
            'sadness', 'surprised',
        ],
        input_channels: int = 1,
        init_out_channels: int = 20,
        final_out_channels: int = 80,
        pool_output_size: Tuple[int] = (1, 46),
        num_layers: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emotions = emotions
        self.input_channels = input_channels
        self.init_out_channels = init_out_channels
        self.final_out_channels = final_out_channels
        self.pool_output_size = pool_output_size
        self.num_layers = num_layers