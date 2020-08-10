from pathlib import Path

import lightgbm
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import shap
import pandas as pd

from Metrics import get_metrics

random_state = 97


def get_path(mode='train'):
    base_path = Path('../data/results/details')

    tag = '200808'
    files = [item.name for item in base_path.glob('*-{}-*{}*'.format(tag, mode))]

    tag = 'haggingface_gru'
    files = [item.name for item in base_path.glob('*-{}-*{}*'.format(tag, mode))] + files

    # files = [
    #     'details-20200801230929-epoch17-{}.csv'.format(mode),
    #     'details-20200801235018-epoch11-{}.csv'.format(mode),
    #     'details-20200801235056-epoch19-{}.csv'.format(mode)
    # ]
    return base_path, files


def get_data(mode='train'):
    base_path, files = get_path(mode=mode)

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



X_train, y_train = get_data(mode='train')
X_test, y_test = get_data(mode='valid')

_, files = get_path('train')
feature_names = files
lgbm = lightgbm.LGBMClassifier(random_state=random_state)
lgbm.fit(y=y_train, X=X_train, feature_name=feature_names)


metrics = get_metrics(predicted_label=lgbm.predict(X=X_test),  labels=y_test)
print(metrics)
print(f1_score(y_true=y_test, y_pred=lgbm.predict(X=X_test)))

with_shap = True
if with_shap:
    shap.initjs()
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(pd.DataFrame(X_train+X_test))
    shap.summary_plot(shap_values, X_train+X_test, plot_type="bar", feature_names=feature_names)

    # 各データの予測に対する各特徴量の寄与
    i = 0
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], (X_train + X_test)[i], feature_names=feature_names)
    shap.force_plot(explainer.expected_value[1], shap_values[1][i], (X_train + X_test)[i], feature_names=feature_names)

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
