from pathlib import Path
from datetime import datetime


def get_now():
    dt_now = datetime.now()
    return dt_now.strftime('%Y.%m.%d %H:%M:%S')


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




if __name__ == "__main__":
    generate_vocabjson()
