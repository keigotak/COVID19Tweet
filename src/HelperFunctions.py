from pathlib import Path
from datetime import datetime

from nltk.corpus import stopwords
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


def get_now():
    dt_now = datetime.now()
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


def get_stop_words():
    return set(stopwords.words('english'))
    # self.stop_words |= set(
    #     ['<hashtag>', '</hashtag>', '<allcaps>', '</allcaps>', '<user>', 'covid19', 'coronavirus', 'covid',
    #      '<number>', 'httpurl', 19, '19'])
    # self.stop_words |= set(["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", "<", ">", "(", ")", "/"])


def get_tokenizer():
    return TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )


if __name__ == "__main__":
    generate_vocabjson()
