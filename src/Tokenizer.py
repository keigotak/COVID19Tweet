from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from TweetNormalizer import normalizeTweet


class Tokenizer:
    __singleton = None
    __tokenizer = None

    def __new__(cls, with_vinai=False):
        if cls.__singleton is None:
            cls.__singleton = super(Tokenizer, cls).__new__(cls)
            if with_vinai:
                cls.__tokenizer = normalizeTweet
            else:
                cls.__tokenizer = TextPreProcessor(
                    # terms that will be normalized
                    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
                    # terms that will be annotated
                    annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
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
                ).pre_process_doc
        return cls.__singleton

    def get_instance(self):
        return self.__tokenizer
Tokenizer()


if __name__ == '__main__':
    tokenizer = Tokenizer().get_instance()
    print(tokenizer)
    print(tokenizer('I have a pen. I have an apple'))

    tokenizer = Tokenizer().get_instance()
    print(tokenizer)
    print(tokenizer('I have a pen. I have a pineapple'))
