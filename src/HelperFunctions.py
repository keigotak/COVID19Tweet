import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch


def set_seed(seed=97):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuda でのRNGを初期化
    torch.cuda.manual_seed(seed)


def get_now(with_path=False):
    dt_now = datetime.now()
    if with_path:
        return dt_now.strftime('%Y%m%d%H%M%S')
    return dt_now.strftime('%Y.%m.%d %H:%M:%S')


def get_milliseconds():
    return datetime.now().strftime('%f')


def get_label_text(num):
    if num == 1:
        return 'INFORMATIVE'
    elif num == 0:
        return 'UNINFORMATIVE'
    else:
        return None


def get_label(text):
    if text == 'INFORMATIVE':
        return 1
    elif text == 'UNINFORMATIVE':
        return 0
    else:
        return None


def get_raw_datasets():
    tags = ['train', 'valid']
    dataset_path = {'train': Path('../data/raw/train.tsv'),
                    'valid': Path('../data/raw/valid.tsv')}
    datasets = {}
    for tag in tags:
        with dataset_path[tag].open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
            headers = texts[0]
            contents = texts[1:]
            datasets[tag] = [line.strip().split('\t') for line in contents]
    return datasets, tags, dataset_path


def get_datasets():
    datasets, tags, dataset_path = get_raw_datasets()
    modified_datasets = {tag: [] for tag in tags}
    for tag in tags:
        for line in datasets[tag]:
            modified_datasets[tag].append([line[1], get_label(line[2])])
    return modified_datasets, tags


def shlink_mergefile():
    with Path("../data/models/BERTweet_base_transformers/bpe.codes").open('r', encoding='utf-8-sig') as f:
        texts = f.readlines()
    with Path("../data/models/BERTweet_base_transformers/bpe.codes.woid").open('w', encoding='utf-8-sig') as f:
        for text in texts:
            line = text.strip().split(' ')[:2]
            f.write(' '.join(line))
            f.write('\n')


def generate_vocabjson():
    with Path("../data/models/BERTweet_base_transformers/dict.txt").open('r', encoding='utf-8-sig') as f:
        texts = f.readlines()
    with Path("../data/models/BERTweet_base_transformers/dict.txt.woid.json").open('w', encoding='utf-8-sig') as f:
        f.write('{')
        f.write('"<pad>": "0",\n')
        f.write('"</s>": "1",\n')
        f.write('"<unk>": "2",\n')
        f.write('"<s>": "3",\n')
        count = 4
        for text in texts:
            word = text.strip().split(' ')[0]
            f.write('"{}": "{}",\n'.format(word, count))
            count += 1


def get_results_path(tag=''):
    return Path('../data/results/results-{}.csv'.format(tag))


def get_details_path(tag=''):
    return Path('../data/results/details/details-{}.csv'.format(tag))


def get_save_model_path(dir_tag, file_tag=''):
    dt_now = datetime.now()
    now = dt_now.strftime('%Y%m%d_%H%M%S')
    directory = Path('../data/results/{}'.format(dir_tag))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / '{}{}.pkl'.format(now, file_tag)


def get_hyperparameter_keys():
    return ['lr',
            'num_head',
            'dropout_ratio',
            'train_batch_size',
            'weight_decay',
            'clip_grad_nurm',
            'optimizer',
            'momentum',
            'model']


if __name__ == "__main__":
    generate_vocabjson()
