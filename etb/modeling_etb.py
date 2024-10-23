""" PyTorch Emotion Transfer Benchmark (ETB) model """

import logging

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .configuration_etb import ETBConfig


class ETBResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool = False):
        super().__init__()
        self.first = first
        self.layers = nn.Sequential()

        if not self.first:
            self.layers.add_module('gnorm1', nn.GroupNorm(in_channels, in_channels))
            self.layers.add_module('leaky_relu1', nn.LeakyReLU(negative_slope=0.3))
        
        self.layers.add_module(
            'conv1', 
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.layers.add_module('gnorm2', nn.GroupNorm(out_channels, out_channels))
        self.layers.add_module('leaky_relu2', nn.LeakyReLU(negative_slope=0.3))
        self.layers.add_module(
            'conv2', 
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        
        self.downsample = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        ) if in_channels != out_channels else None
        
        self.max_pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        outputs = self.layers(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        outputs += residual
        outputs = self.max_pool(outputs)
        return outputs


class ETBSpecRNet(nn.Module):
    def __init__(self, config: ETBConfig):
        super().__init__()
        self.first_gnorm = nn.GroupNorm(num_groups=config.input_channels, num_channels=config.input_channels)
        self.selu = nn.SELU(inplace=True)

        self.block1 = ETBResBlock2D(in_channels=config.input_channels, out_channels=config.init_out_channels, first=True)
        self.block2 = ETBResBlock2D(in_channels=config.init_out_channels, out_channels=config.final_out_channels)
        self.block3 = ETBResBlock2D(in_channels=config.final_out_channels, out_channels=config.final_out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_attention1 = self._attention_fc(in_features=config.init_out_channels, out_features=config.init_out_channels)
        self.fc_attention2 = self._attention_fc(in_features=config.final_out_channels, out_features=config.final_out_channels)
        self.fc_attention3 = self._attention_fc(in_features=config.final_out_channels, out_features=config.final_out_channels)

        self.gnorm_gru = nn.GroupNorm(num_groups=config.final_out_channels, num_channels=config.final_out_channels)
        self.gru = nn.GRU(input_size=config.final_out_channels, hidden_size=config.final_out_channels, num_layers=config.num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(in_features=config.final_out_channels * 2, out_features=config.final_out_channels * 2)
        self.fc2 = nn.Linear(in_features=config.final_out_channels * 2, out_features=1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.adapt = nn.AdaptiveAvgPool2d(config.pool_output_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def _attention_fc(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_gnorm(x)
        x = self.selu(x)

        for block, fc_attention in zip([self.block1, self.block2, self.block3], 
                                       [self.fc_attention1, self.fc_attention2, self.fc_attention3]):
            x = block(x)
            y = self.avg_pool(x).view(x.size(0), -1)
            y = fc_attention(y)
            y = self.sigmoid(y).view(y.size(0), y.size(1), -1, 1)
            x = x * y + y 
            x = self.max_pool(x)

        x = self.gnorm_gru(x)
        x = self.selu(x)
        x = self.adapt(x)
        x = x.squeeze(-2)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)

        return self.sigmoid(x)
    

class ETBPreTrainedModel(PreTrainedModel):
    config_class = ETBConfig
    base_model_prefix = "etb"

    def _init_weights(self, module):
        return super()._init_weights(module)
    

class ETBForEmotionClassification(ETBPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.discriminator = nn.ModuleDict({
            emotion: ETBSpecRNet(config) for emotion in config.emotions
        })

    def forward(self, input_features: torch.Tensor, emotion: str = None) -> torch.Tensor:
        if emotion is not None:
            return self.discriminator[emotion](input_features)
        else:
            return {
                emotion: self.discriminator[emotion](input_features) for emotion in self.config.emotions
            }