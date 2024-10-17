import csv
import os 
import sys
import logging

from librosa import load

import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from transformers import HfArgumentParser
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etb import WavLM, WavLMConfig
from correlation_audio_features.librosa_wrapper import LibrosaWrapper


logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    original_dir: str = field(
        default='../example/original_samples',
        metadata={'help': ''},
    )
    cloned_dir: str = field(
        default='../example/cloned_samples',
        metadata={'help': ''},
    )

def parse_eval_args() -> EvalConfig:
    parser = HfArgumentParser(EvalConfig)
    args = parser.parse_args_into_dataclasses()[0]

    if not args.original_dir:
        raise ValueError('Directory containing original files must be specified.')
    
    if not args.cloned_dir:
        raise ValueError('Directory containing cloned files must be specified.')

    if not os.path.exists(args.original_dir):
        raise ValueError('Selected directory with original files does not exist.')
    
    if not os.path.exists(args.cloned_dir):
        raise ValueError('Selected directory with cloned files does not exist.')
    
    #TODO check if directories are not empty

    #TODO check if there are parallel files in both directories

    logger.info(f'Evaluation arguments: [{args}]')
    
    return args


def main():
    args = parse_eval_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    wrapper = LibrosaWrapper(sampling_rate=16_000)
    # filenames = os.listdir(args.original_dir) # In the case of an error during cloning, cloned samples may be missing
    filenames = os.listdir(args.cloned_dir)

    checkpoint = torch.load('../WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with open('./results.csv', 'w') as fp:
        fp.write('filename,feature,cosine_distance\n')

    for filename in filenames:
        x_orig, _ = load(f'{args.original_dir}/{filename}', sr=16_000)
        x_clon, _ = load(f'{args.cloned_dir}/{filename}', sr=16_000)

        if len(x_orig) < len(x_clon):
            x_clon = x_clon[:len(x_orig)]
        elif len(x_orig) > len(x_clon):
            x_orig = x_orig[:len(x_clon)]

        try:
            original_features = wrapper(waveform=x_orig, sampling_rate=16_000, return_dict=True)
            cloned_features = wrapper(waveform=x_clon, sampling_rate=16_000, return_dict=True)
        except:
            print(f"LibrosaWrapper error, file name: {filename}")
            continue


        all_values = []

        for feature in original_features:
            f_orig, f_clon = original_features[feature], cloned_features[feature]
            f_orig, f_clon = np.nan_to_num(f_orig), np.nan_to_num(f_clon)
            if f_orig.ndim < 2: f_orig = f_orig.reshape(1, -1)
            if f_clon.ndim < 2: f_clon = f_clon.reshape(1, -1)
            val = np.mean(cosine_distances(f_orig, f_clon))
            with open('./results.csv', 'a') as fp:
                fp.write(f'{filename},{feature},{val}\n')
            all_values.append(val)
        
        with open('./results.csv', 'a') as fp:
            fp.write(f'{filename},mean,{np.mean(all_values)}\n')
            fp.write(f'{filename},std,{np.std(all_values)}\n')
            fp.write(f'{filename},min,{np.min(all_values)}\n')
            fp.write(f'{filename},max,{np.max(all_values)}\n')

        x_orig = torch.tensor(x_orig, dtype=torch.float32).view(1, -1).to(device)
        x_clon = torch.tensor(x_clon, dtype=torch.float32).view(1, -1).to(device)
        if cfg.normalize:
            x_orig = torch.nn.functional.layer_norm(x_orig, x_orig.shape)
            x_clon = torch.nn.functional.layer_norm(x_clon, x_clon.shape)
        orig_rep, _ = model.extract_features(x_orig, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        orig_rep = orig_rep.squeeze(0).detach().cpu().numpy()
        clon_rep, _ = model.extract_features(x_clon, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        clon_rep = clon_rep.squeeze(0).detach().cpu().numpy()
        with open('./results.csv', 'a') as fp:
            fp.write(f'{filename},wavlm,{np.mean(cosine_distances(orig_rep, clon_rep))}\n')


if __name__ == '__main__':
    main()