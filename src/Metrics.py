from sklearn.metrics import confusion_matrix


def get_metrics(outputs, labels, is_binary_task=True):

    if is_binary_task:
        tp, fp, fn, tn = get_confusion_matrix(outputs, labels)
    else:
        tp, fp, fn, tn = 0, 0, 0, 0
        rets = confusion_matrix(y_true=labels, y_pred=outputs)
        for i, ret in enumerate(rets):
            for j, item in enumerate(ret):
                if i == j:
                    tp += item
                else:
                    fn += item
                    fp += item

    accuracy = 0.0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def get_confusion_matrix(y_pred, y_true):
    tp, fp, fn, tn = 0, 0, 0, 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
    return tp, fp, fn, tn


def get_print_keys():
    return ['f1', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn']


if __name__ == '__main__':
    # tp: 2, tn: 1, fp:4 fn: 3
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    rets = get_metrics(y_pred, y_true)
    print(rets)
    assert rets['tp'] == 2
    assert rets['tn'] == 1
    assert rets['fp'] == 4
    assert rets['fn'] == 3
