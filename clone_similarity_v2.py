import pandas as pd
import numpy as np

model_name = 'WhisperSpeech'
df_list = {}
df_list['ravdess'] = pd.read_csv(f'/projects/etb/etb/scripts/correlation_audio_features/clone_detection/ravdess/{model_name}/cosine_distance.csv')
df_list['cremad'] = pd.read_csv(f'/projects/etb/etb/scripts/correlation_audio_features/clone_detection/crema-d/{model_name}/cosine_distance.csv')
# df_ravdess = pd.read_csv(f'/projects/etb/etb/scripts/correlation_audio_features/clone_detection/ravdess/{model_name}/cosine_distance.csv')
# df_cremad = pd.read_csv(f'/projects/etb/etb/scripts/correlation_audio_features/clone_detection/crema-d/{model_name}/cosine_distance.csv')

save_path = 'clone_detection'

count = 0
result_df_list = {}
result_df_list['all'] = {}
result_df_list['ravdess'] = {}
result_df_list['cremad'] = {}
emotion_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']


result_df_list['all']['all_emotion'] = {}
for result_df_key in result_df_list.keys():
    if result_df_key == 'all':
        continue
    result_df_list[result_df_key]['all_emotion'] = {}
    for index, sample in df_list[result_df_key].iterrows():
        emotion = sample['filename'].split('_')[1]
        if emotion in emotion_list:
            if emotion not in result_df_list[result_df_key].keys():
                result_df_list[result_df_key][emotion] = {}
            if emotion not in result_df_list['all'].keys():
                result_df_list['all'][emotion] = {}
            # if sample['feature'] not in metric_list:
            if sample['feature'] not in result_df_list[result_df_key][emotion].keys():
                result_df_list[result_df_key][emotion][sample['feature']] = []
            if sample['feature'] not in result_df_list[result_df_key]['all_emotion'].keys():
                result_df_list[result_df_key]['all_emotion'][sample['feature']] = []
            if sample['feature'] not in result_df_list['all'][emotion].keys():
                result_df_list['all'][emotion][sample['feature']] = []
            if sample['feature'] not in result_df_list['all']['all_emotion'].keys():
                result_df_list['all']['all_emotion'][sample['feature']] = []
            result_df_list[result_df_key][emotion][sample['feature']].append(sample['cosine_distance'])
            result_df_list[result_df_key]['all_emotion'][sample['feature']].append(sample['cosine_distance'])
            result_df_list['all'][emotion][sample['feature']].append(sample['cosine_distance'])
            result_df_list['all']['all_emotion'][sample['feature']].append(sample['cosine_distance'])




check = 0
for result_df_key in result_df_list.keys():
    for e in result_df_list[result_df_key].keys():
        for i in result_df_list[result_df_key][e].keys():
            result_df_list[result_df_key][e][i] = np.mean(result_df_list[result_df_key][e][i])

        result_df_list[result_df_key][e]['model'] = f'{model_name}'
        result_df_list[result_df_key][e]['dataset'] = result_df_key
        result_df_list[result_df_key][e]['emotion'] = f'{e}'

    for e in result_df_list[result_df_key].keys():
        df_helper = pd.DataFrame([result_df_list[result_df_key][e]])
        if check == 0:
            df = df_helper.copy()
            check = 1
        else:
            df = pd.concat([df, df_helper], ignore_index=True)



df.to_csv(f'{save_path}/result.csv', index=False)
print(df)