PREDEFINED_DATASETS = {
    'renumics/emodb': {
        'language': 'de',
        'label_mapping': {
            '0': 'anger', '1': 'boredom', '2': 'disgust', '3': 'fear', 
            '4': 'happiness', '5': 'neutral', '6': 'sadness',
        },
    },
    'amu-cai/nEMO': {
        'language': 'pl',
        'label_mapping': {
            'anger': 'anger', 'fear': 'fear', 'happiness': 'happiness',
            'neutral': 'neutral', 'sadness': 'sadness', 'surprised': 'surprised',
        },
    },
    'xbgoose/ravdess': {
        'language': 'en',
        'label_mapping': {
            'angry': 'anger', 'calm': 'calm', 'disgust': 'disgust', 'fearful': 'fear', 
            'happy': 'happiness', 'neutral': 'neutral', 'sad': 'sadness', 'surprised': 'surprised', 
        },
    },
    'sajid73/SUBESCO-audio-dataset': {
        'language': 'bn',
        'columns_to_rename': [
            ['label', 'emotion'],
        ],
        'label_mapping': {
            '0': 'anger', '1': 'disgust', '2': 'fear', '3': 'happiness',
            '4': 'neutral', '5': 'sadness', '6': 'surprised',
        },
    },
    'Pak-Speech-Processing/urdu-emotions': {
        'language': 'ur',
        'columns_to_rename': [
            ['label', 'emotion'],
        ],
        'label_mapping': {
            '0': 'anger', '1': 'happiness', '2': 'neutral',
        },
    },
}