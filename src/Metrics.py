from sklearn.metrics import confusion_matrix


def get_metrics(outputs, labels):
    cm = confusion_matrix(outputs, labels, labels=[0, 1])
    tp, fp, fn, tn = cm.flatten()

    accuracy = 0.0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def get_print_keys():
    return ['f1', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn']