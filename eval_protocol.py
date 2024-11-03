import csv
import os 
import sys
import logging
import pandas as pd
from typing import List

from librosa import load

import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from transformers import HfArgumentParser
from sklearn.metrics.pairwise import cosine_distances


from etb import WavLM, WavLMConfig
from librosa_wrapper import LibrosaWrapper

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    emotion_list: List[str] = field(
        default_factory=list,
        metadata={'help': 'A list of emotions in your dataset'},
    )
    original_dir: str = field(
        default='example/original_samples',
        metadata={'help': 'The path to the folder with original samples'},
    )
    cloned_dir: str = field(
        default='example/cloned_samples',
        metadata={'help': 'The path to the folder with cloned samples'},
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


def data_aggregation(emotion_list):
    data = pd.read_csv('logs.csv')
    #emotion_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
    result_df_list = {}
    result_df_list['all'] = {}
    error_file_name_v1 = ''
    error_file_name_v2 = ''

    for index, sample in data.iterrows():
        if emotion_list == []:
            emotion = None
        else:
            try:
                emotion = sample['filename'].split('_')[-1][:-4]
            except:
                if sample['filename'] != error_file_name_v1:
                    print(f"invalid file name: {sample['filename']}, should be: ..._emotion.wav")
                    error_file_name_v1 = sample['filename']

                emotion = None
        

        if emotion in emotion_list:
            if emotion not in result_df_list.keys():
                result_df_list[emotion] = {}
            if sample['feature'] not in result_df_list[emotion].keys():
                result_df_list[emotion][sample['feature']] = []
            result_df_list[emotion][sample['feature']].append(sample['cosine_distance'])
        elif emotion_list != []:
            if sample['filename'] != error_file_name_v2:
                print(f"invalid emotion name in file: {sample['filename']}, should be: {emotion_list}")
                error_file_name_v2 = sample['filename']

        if sample['feature'] not in result_df_list['all'].keys():
            result_df_list['all'][sample['feature']] = []    
        result_df_list['all'][sample['feature']].append(sample['cosine_distance'])


    for emotion in result_df_list.keys():
        for feature in result_df_list[emotion].keys():
            result_df_list[emotion][feature] = np.mean(result_df_list[emotion][feature])

        #result_df_list[e]['model'] = f'{model_name}'
        #result_df_list[e]['dataset'] = f'{dataset_name}'
        result_df_list[emotion]['emotion'] = f'{emotion}'

    df = pd.DataFrame(columns=result_df_list['all'].keys())
    for emotion in result_df_list.keys():
        df_helper = pd.DataFrame([result_df_list[emotion]])

        df = pd.concat([df, df_helper], ignore_index=True)
    
    df.to_csv('results.csv', index=False)
    print()
    print(df)


def main():
    args = parse_eval_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    wrapper = LibrosaWrapper(sampling_rate=16_000)
    # filenames = os.listdir(args.original_dir) # In the case of an error during cloning, cloned samples may be missing
    filenames = os.listdir(args.cloned_dir)

    checkpoint = torch.load('WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with open('./logs.csv', 'w') as fp:
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
            with open('./logs.csv', 'a') as fp:
                fp.write(f'{filename},{feature},{val}\n')
            all_values.append(val)
        
        with open('./logs.csv', 'a') as fp:
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
        with open('./logs.csv', 'a') as fp:
            fp.write(f'{filename},wavlm,{np.mean(cosine_distances(orig_rep, clon_rep))}\n')

    data_aggregation(args.emotion_list)



        


if __name__ == '__main__':
    main()