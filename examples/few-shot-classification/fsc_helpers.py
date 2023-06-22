from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
import datasets
from fsc_reward import PromptedClassificationReward


class PromptedClassificationDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        class_labels: List[str],
        source_2_texts: List[str]
    ):
        assert len(source_texts) == len(class_labels)
        self.source_texts = source_texts
        self.class_labels = class_labels
        self.source_2_texts = source_2_texts

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'class_labels': self.class_labels[idx],
                'source_2_texts': self.source_2_texts[idx]}
        return item


def make_few_shot_classification_dataset(
        config: "DictConfig") -> Tuple[PromptedClassificationDataset]: 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, class_labels, num_classes, verbalizers, template, source_2_texts = \
            load_few_shot_classification_dataset(config.dataset, 
                                                 config.dataset_seed, 
                                                 split, config.base_path, 
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                    class_labels, source_2_texts)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'],
            num_classes, verbalizers, template)


def get_data(dataset_name, data) -> tuple:
    if 'xnli' in dataset_name or dataset_name == 'mnli' or 'anli' in dataset_name or 'americas_nli' in dataset_name or dataset_name == 'snli':
        return [d["premise"] for d in data], [d["hypothesis"] for d in data], [d["label"] for d in data]
    elif dataset_name == 'sst2':
        return [d["sentence"] for d in data], [d["sentence"] for d in data], [d["label"] for d in data]
    elif dataset_name == 'rte' or dataset_name == 'mrpc':
        return [d["sentence1"] for d in data], [d["sentence2"] for d in data], [d["label"] for d in data]
    elif dataset_name == 'qnli':
        return [d["question"] for d in data], [d["sentence"] for d in data], [d["label"] for d in data]
    elif dataset_name == 'qqp':
        return [d["question1"] for d in data], [d["question2"] for d in data], [d["label"] for d in data]
    elif dataset_name == 'boolq':
        return [d["question"] for d in data], [d["passage"] for d in data], [d["label"] for d in data]
    elif 'indonlp/NusaX-senti' in dataset_name or dataset_name == 'yelp_polarity' or dataset_name == 'ag_news' or dataset_name == 'dbpedia_14' or dataset_name == 'SetFit/sst5':
        return [d["text"] for d in data], [d["text"] for d in data], [d["label"] for d in data]


def load_few_shot_classification_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int
) -> Tuple[List[str]]:
    # assert dataset in ['agnews', 'cr', 'mr', 'sst-2', 
    #                    'sst-5', 'yelp-2', 'yelp-5']
    # assert split in ['train', 'dev', 'test']
    # assert num_shots in [16]

    # seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
    # seed_path = seed_dict[dataset_seed]
    # filepath = f'{num_shots}-shot/{dataset}/{seed_path}/{split}.tsv'
    # full_filepath = os.path.join(base_path, filepath)
    # df = pd.read_csv(full_filepath, sep='\t')
    # if 'text' in df:
    #     source_texts = df.text.tolist()
    # else: 
    #     source_texts = df.sentence.tolist()
    # class_labels = df.label.tolist()
    glue_list = ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']
    superglue_list = ['boolq']
    if 'xnli' in dataset:
        sub_split = dataset.split('_')[1]
        data = datasets.load_dataset('xnli', sub_split)
    elif 'americas_nli' in dataset:
        sub_split = dataset.split('_')[2]
        data = datasets.load_dataset('americas_nli', sub_split)
    elif dataset in glue_list:
        data = datasets.load_dataset('glue', dataset)
    elif 'anli' in dataset:
        data = datasets.load_dataset('anli')
    elif dataset in superglue_list:
        data = datasets.load_dataset('super_glue', dataset)
    elif 'indonlp/NusaX-senti' in dataset:
        sub_split = dataset.split('_')[1]
        data = datasets.load_dataset('indonlp/NusaX-senti', sub_split)
    else:
        data = datasets.load_dataset(dataset)
    
    if dataset == 'mnli':
        train_dataset = data['train']
        val_dataset = data['validation_matched']
        test_dataset = data['test_matched']
    elif 'anli' in dataset:
        split = dataset.split('_')[1]
        train_dataset = data['train_'+split]
        val_dataset = data['dev_'+split]
        test_dataset = data['test_'+split]
    elif 'americas_nli' in dataset:
        train_dataset = data['validation']
        val_dataset = data['validation']
        test_dataset = data['test']
    elif dataset == 'yelp_polarity' or dataset == 'ag_news':
        train_dataset = data['train']
        val_dataset = data['train']
        test_dataset = data['test']
    elif dataset == 'snli':
        train_dataset = [x for x in data['train'] if x['label'] != -1]
        val_dataset = [x for x in data['validation'] if x['label'] != -1]
        test_dataset = [x for x in data['test'] if x['label'] != -1]
    else:
        train_dataset = data['train']
        val_dataset = data['validation']
        test_dataset = data['test']
    train_0 = [x for x in train_dataset if x['label'] == 0][:num_shots]
    train_1 = [x for x in train_dataset if x['label'] == 1][:num_shots]
    train_2 = [x for x in train_dataset if x['label'] == 2][:num_shots]
    train_3 = [x for x in train_dataset if x['label'] == 3][:num_shots]
    train_dataset = train_0 + train_1 + train_2 + train_3
    flag = False
    if dataset in glue_list:
        val_0 = [x for x in train_dataset if x['label'] == 0][-num_shots:]
        val_1 = [x for x in train_dataset if x['label'] == 1][-num_shots:]
        val_2 = [x for x in train_dataset if x['label'] == 2][-num_shots:]
        new_val_dataset = val_0 + val_1 + val_2
        test_dataset = val_dataset
    elif dataset == 'ag_news':
        val_0 = [x for x in train_dataset if x['label'] == 0][-num_shots:]
        val_1 = [x for x in train_dataset if x['label'] == 1][-num_shots:]
        val_2 = [x for x in train_dataset if x['label'] == 2][-num_shots:]
        val_3 = [x for x in train_dataset if x['label'] == 3][-num_shots:]
        new_val_dataset = val_0 + val_1 + val_2 + val_3
        test_dataset = val_dataset
    else:
        val_0 = [x for x in val_dataset if x['label'] == 0][:num_shots]
        val_1 = [x for x in val_dataset if x['label'] == 1][:num_shots]
        val_2 = [x for x in val_dataset if x['label'] == 2][:num_shots]
        val_dataset = val_0 + val_1 + val_2
        flag = True
    if split == 'train':
        source_texts, source_2_texts, class_labels = get_data(dataset, train_dataset)
    elif split == 'dev':
        if not flag:
            source_texts, source_2_texts, class_labels = get_data(dataset, new_val_dataset)
        else:
            source_texts, source_2_texts, class_labels = get_data(dataset, val_dataset)
    elif split == 'test':
        source_texts, source_2_texts, class_labels = get_data(dataset, test_dataset)
    verbalizers = get_dataset_verbalizers(dataset)
    num_classes = len(verbalizers)

    template = None

    return (source_texts, class_labels, 
            num_classes, verbalizers, template, source_2_texts)


def get_dataset_verbalizers(dataset: str) -> List[str]: 
    # verbalizers = None
    # if dataset in ['sst2', 'yelp-2', 'mr', 'cr']:
    #     verbalizers = ['▁negative', '▁positive'] # num_classes
    # elif dataset == 'agnews': 
    #     verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
    # elif dataset in ['sst-5', 'yelp-5']:
    #     verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', 
    #                    '\u0120good', '\u0120great'] # num_classes
    # elif dataset == 'subj':
    #     verbalizers = ['\u0120subjective', '\u0120objective']
    # elif dataset == 'trec':
    #     verbalizers = ['\u0120Description', '\u0120Entity',
    #                 '\u0120Expression', '\u0120Human',
    #                 '\u0120Location', '\u0120Number']
    # elif dataset == 'yahoo':
    #     verbalizers = ['culture', 'science',
    #                 'health', 'education',
    #                 'computer', 'sports',
    #                 'business', 'music',
    #                 'family', 'politics']
    # elif dataset == 'dbpedia':
    #     verbalizers = ['\u0120Company', '\u0120Education',
    #                 '\u0120Artist', '\u0120Sports',
    #                 '\u0120Office', '\u0120Transportation',
    #                 '\u0120Building', '\u0120Natural',
    #                 '\u0120Village', '\u0120Animal',
    #                 '\u0120Plant', '\u0120Album',
    #                 '\u0120Film', '\u0120Written']
    if 'xnli' in dataset or dataset == 'mnli' or 'anli' in dataset or 'americas_nli' in dataset or dataset == 'snli':
        verbalizer_predefined = ['yes', 'maybe', 'no']
        # verbalizer_predefined = ['yes', 'neither', 'no']
    elif dataset == 'sst2' or dataset == 'yelp_polarity':
        # verbalizer_predefined = ['terrible', 'great']
        verbalizer_predefined = ['negative', 'positive']
    elif dataset == 'rte' or dataset == 'qnli':
        verbalizer_predefined = ['yes', 'no']
    elif dataset == 'mrpc' or dataset == 'qqp':
        verbalizer_predefined = ['no', 'yes']
    elif dataset == 'boolq':
        verbalizer_predefined = ['no', 'yes']
    elif 'indonlp/NusaX-senti' in dataset:
        verbalizer_predefined = ['negative', 'neutral', 'positive']
    elif dataset == 'ag_news':
        verbalizer_predefined = ['World', 'Sports', 'Business', 'Technology']
    elif dataset == 'dbpedia_14':
        verbalizer_predefined = ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder', 'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'WrittenWork']
    elif dataset == 'SetFit/sst5':
        verbalizer_predefined = ['terrible', 'bad', 'okay', 'good', 'great']
    special_space = '▁'
    # special_space = 'Ġ'
    verbalizer_predefined = [special_space + v for v in verbalizer_predefined]
    
    return verbalizer_predefined


@dataclass
class FewShotClassificationDatasetConfig:
    dataset: str = "???"
    dataset_seed: Optional[int] = None 
    base_path: str = './data'
    num_shots: int = 16


def make_prompted_classification_reward(
    num_classes: int,
    verbalizers: List[str],
    template: Optional[str],  
    config: "DictConfig") -> PromptedClassificationReward:
    return PromptedClassificationReward(config.task_lm, config.is_mask_lm, 
                                        config.compute_zscore, 
                                        config.incorrect_coeff, 
                                        config.correct_coeff,
                                        num_classes, verbalizers, template, config.dataset)


@dataclass
class PromptedClassificationRewardConfig:
    task_lm: str = 'distilroberta-base'
    is_mask_lm: Optional[bool] = None
    compute_zscore: bool = True
    incorrect_coeff: float = 180.0
    correct_coeff: float = 200.0
