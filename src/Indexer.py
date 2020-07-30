from StopWords import StopWords
from Tokenizer import Tokenizer


class SentenceIndexer:
    __singleton = None
    __sentence2indexes = None
    __indexes2sentence = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = super(SentenceIndexer, cls).__new__(cls)
            cls.__sentence2indexes = {}
            cls.__indexes2sentence = {}
        return cls.__singleton

    def get_instance(self):
        return self.__sentence2indexes, self.__indexes2sentence


class Indexer:
    def __init__(self, special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_preprocess=True, lower_count=10):
        if special_tokens is None:
            self.word2index = {'<unk>': 0, '<pad>': 1}
            self.current = 2
        else:
            self.word2index = special_tokens
            self.current = len(special_tokens)
        self.index2word = {val: key for key, val in special_tokens.items()}
        self.vocab = set([key for key, val in special_tokens.items()])

        self.sentence2indexes, self.indexes2sentence = SentenceIndexer().get_instance()

        self.padding_index = self.word2index['<pad>']
        self.unknown_index = self.word2index['<unk>']

        self.with_preprocess = with_preprocess
        self.delim = ' '
        self.counts = {}
        self.lower_count = lower_count
        self.max_length = 0

        self.stop_words = StopWords().get_instance()
        self.text_processor = Tokenizer().get_instance()

    def __len__(self):
        return self.current

    def tokenize(self, sentence):
        if self.with_preprocess:
            sentence = [word for word in self.text_processor(sentence) if word not in self.stop_words]
        else:
            sentence = sentence.strip().split(' ')
        return sentence

    def count_word(self, word):
        if word not in self.counts.keys():
            self.counts[word] = 1
        else:
            self.counts[word] += 1

    def count_word_in_sentence(self, sentence):
        for word in self.tokenize(sentence):
            self.count_word(word)

    def count_word_in_text(self, text):
        for sentence in text:
            self.count_word_in_sentence(sentence)

    def add_word(self, word):
        if self.with_preprocess:
            # if word in emoji.UNICODE_EMOJI:
            #     print(word)
            if word in self.counts.keys() and self.counts[word] < self.lower_count:
                return
        if word not in self.vocab:
            self.word2index[word] = self.current
            self.index2word[self.current] = word
            self.vocab.add(word)
            self.current += 1

    def add_sentence(self, sentence, with_raw=False):
        if with_raw:
            for word in sentence:
                self.add_word(word)
        else:
            for word in self.tokenize(sentence):
                self.add_word(word)

    def add_sentences(self, sentences, with_raw=False):
        for sentence in sentences:
            self.add_sentence(sentence, with_raw=with_raw)

    def get_index(self, word):
        if word in self.vocab:
            return self.word2index[word]
        else:
            return self.unknown_index

    def sentence_to_index(self, raw_sentence, with_raw=False):
        if raw_sentence in self.sentence2indexes.keys():
            return self.sentence2indexes[raw_sentence]

        if not with_raw:
            sentence = self.tokenize(raw_sentence)
        indexes = [self.get_index(word) for word in sentence]
        self.sentence2indexes[raw_sentence] = indexes
        self.indexes2sentence[' '.join(map(str, indexes))] = [sentence, raw_sentence]
        return indexes

    def text_to_index(self, text, with_raw=False):
        return [self.sentence_to_index(sentence, with_raw=with_raw) for sentence in text]

    def get_word(self, index):
        if index in self.index2word.keys():
            return self.index2word[index]
        else:
            return self.index2word[self.unknown_index]

    def sentence_to_words(self, indexes):
        return [self.get_word(index) for index in indexes]

    def text_to_words(self, index_text):
        return [self.sentence_to_words(indexes) for indexes in index_text]


if __name__ == '__main__':
    indexer = Indexer()
    from HelperFunctions import get_datasets
    datasets, tags = get_datasets()
    sentences = [pairs[0] for pairs in datasets['train']]
    indexer.count_word_in_text(sentences)
    indexer.add_sentences(sentences)
    sentences = [pairs[0] for pairs in datasets['valid']]
    indexes = indexer.text_to_index(sentences)
    unk_count = 0
    for sentence, index in zip(sentences, indexes):
        for word, id in zip(sentence.split(' '), index):
            if id == indexer.unknown_index:
                print('{}({})'.format(word, id))
                unk_count += 1
    print(unk_count)
    for i in range(10):
        print(indexer.sentence_to_words(indexes[i]))
