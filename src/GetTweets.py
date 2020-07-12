import json
import tweepy
from pathlib import Path
import pickle


def get_api():
    with Path('../data/apis/apis.json').open('r') as f:
        CONFIG = json.load(f)
    CONSUMER_KEY = CONFIG["key"]
    CONSUMER_SECRET = CONFIG["skey"]
    ACCESS_TOKEN = CONFIG["access_token"]
    ACCESS_SECRET = CONFIG["saccess_token"]

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api


def get_path(mode):
    if mode == 'train':
        return Path('../data/raw/train.tsv')
    elif mode == 'valid':
        return Path('../data/raw/valid.tsv')
    else:
        return None


def get_texts(path):
    with path.open('r', encoding='utf-8-sig') as f:
        texts = f.readlines()
    headers = texts[0]
    contents = texts[1:]
    texts = [content.strip().split('\t') for content in contents]
    return texts


def get_tweet_status(api, tweet_id):
    if type(tweet_id) is not int:
        tweet_id = int(tweet_id)
    try:
        tweet = api.get_status(tweet_id)
    except tweepy.error.TweepError:
        tweet = None
    return tweet


def get_tweets():
    api = get_api()
    tweets = {}
    for mode in ['train', 'valid']:
        if mode not in tweets.keys():
            tweets[mode] = {}
        path = get_path(mode=mode)
        texts = get_texts(path)
        for text in texts:
            tweet = get_tweet_status(api, text[0])
            tweets[mode][text[0]] = [tweet, text[1], text[2]]

    with Path('../data/analytic/tweets.pkl').open('wb') as f:
        pickle.dump(tweets, f)


def open_pickle():
    with Path('../data/analytic/tweets.pkl').open('rb') as f:
        tweets = pickle.load(f)
    return tweets



if __name__ == '__main__':
    reset_dumps = False
    if reset_dumps:
        get_tweets()
    else:
        tweets = open_pickle()
        print(tweets)