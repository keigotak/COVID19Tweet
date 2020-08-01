from pathlib import Path

import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import shap
import pandas as pd

from Metrics import get_metrics

random_state = 97


def get_path():
    base_path = Path('../data/results/details')
    files = [
        'details-20200731220126-epoch1.csv',
        'details-20200731220448-epoch1.csv',
        'details-20200731220708-epoch1.csv'
    ]
    return base_path, files


def get_data():
    base_path, files = get_path()

    sentences = []
    labels = {file: {} for file in files}
    preds = {file: {} for file in files}
    sentence_flg = True
    for file in files:
        with (base_path / file).open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
        for text in texts:
            items = text.strip().split('\t')
            if sentence_flg:
                sentences.append(items[0])
            labels[file][items[0]] = items[1]
            preds[file][items[0]] = items[3]
        sentence_flg = False

    xs = []
    ys = []
    for sentence in sentences:
        ys.append(int(labels[files[0]][sentence]))
        x = []
        for file in files:
            if sentence not in preds[file].keys():
                x.append(0.0)
            else:
                x.append(float(preds[file][sentence]))
        xs.append(x)

    return xs, ys



xs, ys = get_data()
feature_names = ['model1', 'model2', 'model3']

X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.33, random_state=random_state)
lgbm = lightgbm.LGBMClassifier(random_state=random_state)
lgbm.fit(y=y_train, X=X_train, feature_name=feature_names)


metrics = get_metrics(lgbm.predict(X=X_test),  y_test)
print(metrics)
print(f1_score(y_true=y_test, y_pred=lgbm.predict(X=X_test)))

with_shap = True
if with_shap:
    shap.initjs()
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(pd.DataFrame(X_test))
    shap.summary_plot(shap_values, xs, plot_type="bar", feature_names=feature_names)

with_importance = True
if with_importance:
    # 分割に登場した回数を元に特徴量重要度を算出
    lightgbm.plot_importance(lgbm, importance_type='split', figsize=(5, 9))
    plt.show()

    # 情報利得を元に特徴量重要度を算出
    lightgbm.plot_importance(lgbm, importance_type='gain', figsize=(5, 9))
    plt.show()

    # permutation importanceを算出
    result = permutation_importance(lgbm, X_test, y_test, scoring=make_scorer(f1_score), n_repeats=100,
                                    random_state=random_state)

    pd.Series(result['importances_mean'], index=feature_names).sort_values().plot.barh(figsize=(5, 9))
    plt.ylabel('Features')
    plt.xlabel('Permutation importance')
    plt.vlines(x=0, ymin=-100, ymax=100, linewidth=0.6)
    plt.grid(alpha=0.7)
    plt.show()