import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('../data/models/glove.twitter.27B/glove.twitter.27B.200d.txt')

print(model)

