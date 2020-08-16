import os
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, ElasticNetCV, LassoCV, LassoLarsCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from scipy.sparse import hstack

from HelperFunctions import get_now

class_names = ['INFORMATIVE', 'UNINFORMATIVE']

train = pd.read_table('../data/raw/train.tsv', sep='\t').fillna(' ')
test = pd.read_table('../data/raw/valid.tsv', sep='\t').fillna(' ')

train_text = train['Text']
test_text = test['Text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 5),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'Id': test['Id']})

train['Label'] = train['Label'].mask(train['Label'] == 'INFORMATIVE', int(1))
train['Label'] = train['Label'].mask(train['Label'] == 'UNINFORMATIVE', int(0))
train_target = train['Label'].tolist()

test['Label'] = test['Label'].mask(test['Label'] == 'INFORMATIVE', int(1))
test['Label'] = test['Label'].mask(test['Label'] == 'UNINFORMATIVE', int(0))
test_target = test['Label'].tolist()


def my_f1_score(estimator, x, y):
    yPred = estimator.predict(x)
    return f1_score(y, yPred, pos_label=1, labels=[0, 1])

n_jobs = 20
random_state = 97
max_iter = 10**3
classifiers = {
    # 'logisticregression': LogisticRegression(C=0.1, solver='sag', n_jobs=n_jobs, random_state=random_state, max_iter=max_iter),
    # 'lasso': LassoCV(cv=3, random_state=random_state, n_jobs=n_jobs, alphas=10 ** np.arange(-6, 1, 0.1)),
    # 'lassolars': LassoLarsCV(cv=3, n_jobs=n_jobs),
    # 'multitasklasso': MultiTaskLassoCV(cv=3, random_state=random_state, n_jobs=n_jobs),
    'ridge': RidgeClassifierCV(cv=3),
    'elasticnet': ElasticNetCV(cv=3, random_state=random_state, n_jobs=n_jobs),
    # 'multitaskelasticnet': MultiTaskElasticNetCV(cv=3, random_state=random_state, n_jobs=n_jobs),
    'SVM': SVC(probability=True, random_state=random_state),
    'xgbt': lgb
}

for name, classifier in classifiers.items():
    now, now_dir = get_now()
    print('[{}] start: {}'.format(now, name))
    if name in {'logisticregression'}:
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring=my_f1_score))
        classifier.fit(train_features, train_target)
        scores.append(cv_score)
        print('CV score is {}'.format(cv_score))
        preds = classifier.predict_proba(test_features)[:, 1]
    elif name in {'lassolars'}:
        classifier.fit(train_features.todense(), train_target)
        preds = classifier.predict(test_features)
    elif name in {'xgbt'}:
        lgb_train = lgb.Dataset(train_features, label=train_target)
        lgb_eval = lgb.Dataset(test_features, label=test_target, reference=lgb_train)
        params = {"max_depth": 5, "num_leaves": 10, "learning_rate": 0.001, "bagging_freq": 4, "num_iteration": 100,
                  "bagging_fraction": 0.7, "feature_fraction": 0.6, "max_bin": 63, "lambda_l1": 0, "lambda_l2": 0,
                  "min_data_in_leaf": 100, 'metric': 'f1score'}
        print(params)

        from sklearn.metrics import f1_score

        def lgb_f1_score(y_hat, data):
            y_true = data.get_label()
            y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
            return 'f1', f1_score(y_true, y_hat), True

        gbm = lgb.train(params,
                        lgb_train,
                        feval=lgb_f1_score,
                        num_boost_round=20,
                        valid_sets=[lgb_eval, lgb_train],
                        early_stopping_rounds=5)
        preds = gbm.predict(test_features, num_iteration=gbm.best_iteration)
    else:
        classifier.fit(train_features, train_target)
        preds = classifier.predict(test_features)

    labeled_preds = [0 if item < 0.5 else 1 for item in preds.tolist()]
    valid_score = f1_score(y_pred=labeled_preds, y_true=test_target)
    print('Valid score is {}'.format(valid_score))

    os.makedirs('../data/results/linear', exist_ok=True)
    submission = pd.DataFrame([[text, pred] for text, pred in zip(test['Text'].tolist(), preds.tolist())])
    submission.to_csv('../data/results/linear/{}-ngram.csv'.format(name), sep='\t', index=False, header=False)

    now, now_dir = get_now()
    print('[{}] finished: {}'.format(now, name))

