import logging 

from datasets import (
    Audio,
    Dataset, 
    Value,
    load_dataset, 
    concatenate_datasets
)

from .utils_etb import PREDEFINED_DATASETS


logger = logging.getLogger(__name__)


class ETBDataset:
    """
    A dataset class for loading and processing datasets for Emotion Transfer Benchmarking.

    Attributes:
        data (Dataset): The aggregated and filtered dataset.
        emotions (list[str]): The list of emotions to be considered in the dataset.

    Parameters:
        datasets (list[str]): A list of dataset names to be loaded.
        emotions (list[str], optional): A list of specific emotions to filter the datasets by. If None,
            all available emotions in the datasets are used.
        streaming (bool, optional): Indicates whether the datasets should be loaded in streaming mode.
            Defaults to False.
    """
    def __init__(
        self, 
        datasets: list[str], 
        emotions: list[str],
        seed: int,
        streaming: bool = False,
    ):
        """Initializes the ETBDataset with the specified datasets, optional emotion filtering, and streaming mode."""
        super(ETBDataset, self).__init__()
        self.train_data = self.load_data(datasets, streaming, seed)
        if emotions is None:
            logger.info('Emotions not specified. Obtaining list of available emotions from dataset.')
            self.emotions = self.get_available_emotions(self.train_data)
        else:
            self.emotions = emotions
            self.train_data = self.filter_emotions(dataset=self.train_data, emotions=emotions)
        logger.info(f'Following emotions will be used: {self.emotions}')

    def filter_emotions(self, dataset: Dataset, emotions: list[str]) -> Dataset:
        """Filters the dataset to only include examples with specified emotions."""
        logger.info(f'Limiting dataset to emotions: {emotions}')
        return dataset.filter(lambda example: example['emotion'] in emotions)

    def get_available_emotions(self, dataset: Dataset) -> list:
        """Extracts and returns a list of unique emotions available in the dataset."""
        return list(set(dataset['emotion']))

    def process_dataset(self, dataset_name: str, streaming: bool) -> Dataset:
        """
        Processes the specified dataset, applying predefined transformations based on dataset name,
        and returns the processed dataset.
        """
        logger.info(f'Processing {dataset_name} dataset.')
        dataset = load_dataset(dataset_name, split='train', streaming=streaming)
        if dataset_name in PREDEFINED_DATASETS:
            logger.info(f'Using predefined metadata for {dataset_name} dataset.')
            dataset_info = PREDEFINED_DATASETS[dataset_name]
            if 'columns_to_rename' in dataset_info:
                for old_name, new_name in dataset_info['columns_to_rename']:
                    logger.info(f'{dataset_name}: Renaming column {old_name} to {new_name}.')
                    dataset = dataset.rename_column(old_name, new_name)
            if 'label_mapping' in dataset_info:
                logger.info(f'{dataset_name}: Applying label mapping.')
                dataset = dataset.cast_column('emotion', Value(dtype='string'))
                dataset = dataset.map(self.apply_label_mapping, fn_kwargs={'label_mapping': dataset_info['label_mapping']})
        else:
            logger.info(f'{dataset_name} dataset not found in predefined. Default processing applied.')
            dataset = dataset.cast_column('emotion', Value(dtype='string'))
        logger.info(f"{dataset_name}: Removing columns: {[col for col in dataset.column_names if col not in {'audio', 'emotion'}]}")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in {'audio', 'emotion'}])
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
        dataset = dataset.add_column('dataset_name', [dataset_name] * len(dataset))
        logger.info(f'Found total of {len(dataset)} in {dataset_name} dataset.')
        return dataset

    def apply_label_mapping(self, example: dict, label_mapping: dict) -> dict:
        """Applies a mapping to the emotion labels in the dataset based on the provided label mapping."""
        example['emotion'] = label_mapping[example['emotion']]
        return example
    
    def load_data(self, datasets: list[str], streaming: bool, seed: int) -> Dataset:
        """
        Aggregates data from specified datasets, applying necessary preprocessing,
        filtering by emotions if specified, and preparing for usage in benchmarking.
        """
        dataset = []
        for dataset_name in datasets:
            current_dataset = self.process_dataset(dataset_name, streaming)
            dataset.append(current_dataset)
        logger.info(f'Merging datasets: {datasets}')
        dataset = concatenate_datasets(dataset)
        logger.info(f'Shuffling dataset with seed = {seed}')
        dataset = dataset.shuffle(seed=seed)
        return dataset