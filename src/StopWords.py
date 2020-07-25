from nltk.corpus import stopwords as nltk_stopwords


class StopWords:
    __singleton = None
    __simple_stopwords = None
    __applied_stopwords = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            cls.__singleton = super(StopWords, cls).__new__(cls)
            cls.__simple_stopwords = set(nltk_stopwords.words('english'))
            cls.__applied_stopwords = set(nltk_stopwords.words('english'))
            cls.__applied_stopwords |= {'<hashtag>', '</hashtag>', '<allcaps>', '</allcaps>', '<user>', 'covid19',
                                       'coronavirus', 'covid', '<number>', 'httpurl', 19, '19'}
            cls.__applied_stopwords |= {"'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", "<", ">", "(", ")", "/"}
        return cls.__singleton

    def get_instance(self, with_applied=False):
        if not with_applied:
            return self.__applied_stopwords
        return self.__simple_stopwords


if __name__ == '__main__':
    stopwords = StopWords().get_instance()
    print(id(stopwords))
    print(stopwords)

    stopwords = StopWords().get_instance()
    print(id(stopwords))
    print(stopwords)

    stopwords = StopWords().get_instance(with_applied=True)
    print(id(stopwords))
    print(stopwords)

    stopwords = StopWords().get_instance(with_applied=True)
    print(id(stopwords))
    print(stopwords)
