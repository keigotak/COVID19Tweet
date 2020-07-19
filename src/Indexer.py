from nltk.corpus import stopwords
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from TweetNormalizer import normalizeTweet

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

        self.sentence2indexes = {}
        self.indexes2sentence ={}

        self.padding_index = self.word2index['<pad>']
        self.unknown_index = self.word2index['<unk>']

        self.with_preprocess = with_preprocess
        self.delim = ' '
        self.counts = {}
        self.lower_count = lower_count
        self.max_length = 0

        self.stop_words = set(stopwords.words('english'))
        # self.stop_words |= set(
        #     ['<hashtag>', '</hashtag>', '<allcaps>', '</allcaps>', '<user>', 'covid19', 'coronavirus', 'covid',
        #      '<number>', 'httpurl', 19, '19'])
        # self.stop_words |= set(["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", "<", ">", "(", ")", "/"])

        # self.text_processor = TextPreProcessor(
        #     # terms that will be normalized
        #     normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        #                'time', 'url', 'date', 'number'],
        #     # terms that will be annotated
        #     annotate={"hashtag", "allcaps", "elongated", "repeated",
        #               'emphasis', 'censored'},
        #     # annotate={"elongated", "repeated",
        #     #           'emphasis', 'censored'},
        #     fix_html=True,  # fix HTML tokens
        #
        #     # corpus from which the word statistics are going to be used
        #     # for word segmentation
        #     segmenter="twitter",
        #
        #     # corpus from which the word statistics are going to be used
        #     # for spell correction
        #     corrector="twitter",
        #
        #     unpack_hashtags=True,  # perform word segmentation on hashtags
        #     unpack_contractions=True,  # Unpack contractions (can't -> can not)
        #     spell_correct_elong=False,  # spell correction for elongated words
        #
        #     # select a tokenizer. You can use SocialTokenizer, or pass your own
        #     # the tokenizer, should take as input a string and return a list of tokens
        #     tokenizer=SocialTokenizer(lowercase=True).tokenize,
        #
        #     # list of dictionaries, for replacing tokens extracted from the text,
        #     # with other expressions. You can pass more than one dictionaries.
        #     dicts=[emoticons]
        # )
        self.text_processor = normalizeTweet

    def __len__(self):
        return self.current

    def tokenize(self, sentence):
        if self.with_preprocess:
            # sentence = [word for word in self.text_processor.pre_process_doc(sentence) if word not in self.stop_words]
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

    def add_sentences(self, sentences):
        for sentence in sentences:
            self.add_sentence(sentence)

    def get_index(self, word):
        if word in self.vocab:
            return self.word2index[word]
        else:
            return self.unknown_index

    def sentence_to_index(self, raw_sentence, with_raw=False):
        joined_raw_sentence = ' '.join(raw_sentence)
        if joined_raw_sentence in self.sentence2indexes.keys():
            return self.sentence2indexes[joined_raw_sentence]

        if not with_raw:
            sentence = self.tokenize(raw_sentence)
        indexes = [self.get_index(word) for word in sentence]
        self.sentence2indexes[joined_raw_sentence] = indexes
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
