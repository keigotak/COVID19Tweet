from HelperFunctions import get_raw_datasets
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def probe_labels():
    datasets, tags, path = get_raw_datasets()
    labels = {tag: set() for tag in tags}
    counts = {tag: {} for tag in tags}

    for tag in tags:
        for id, text, label in datasets[tag]:
            labels[tag].add(label)
            if label in counts[tag].keys():
                counts[tag][label] += 1
            else:
                counts[tag][label] = 1
    return {'counts': counts, 'labels': labels}


def probe_sentence_length():
    datasets, tags, path = get_raw_datasets()
    counts = {tag: {} for tag in tags}
    counts_by_labels = {tag: {'INFORMATIVE': {}, 'UNINFORMATIVE': {}} for tag in tags}

    for tag in tags:
        for id, text, label in datasets[tag]:
            words = text.split(' ')
            if len(words) in counts[tag].keys():
                counts[tag][len(words)].append(words)
            else:
                counts[tag][len(words)] = [words]

            if len(words) in counts_by_labels[tag][label].keys():
                counts_by_labels[tag][label][len(words)] += 1
            else:
                counts_by_labels[tag][label][len(words)] = 1

    return {'counts': counts, 'counts_by_labels': counts_by_labels}


def probe_vocabs():
    datasets, tags, path = get_raw_datasets()
    vocabs = {tag: set() for tag in tags}
    counts = {tag: {} for tag in tags}
    vocabs_by_labels = {tag: {'INFORMATIVE': set(), 'UNINFORMATIVE': set()} for tag in tags}
    counts_by_labels = {tag: {'INFORMATIVE': {}, 'UNINFORMATIVE': {}} for tag in tags}

    for tag in tags:
        for id, text, label in datasets[tag]:
            words = text.split(' ')
            for word in words:
                word = word.lower()
                vocabs[tag].add(word)
                vocabs_by_labels[tag][label].add(word)
                if word in counts[tag].keys():
                    counts[tag][word] += 1
                else:
                    counts[tag][word] = 1
                if word in counts_by_labels[tag][label].keys():
                    counts_by_labels[tag][label][word] += 1
                else:
                    counts_by_labels[tag][label][word] = 1
    return {'counts': counts, 'vocabs': vocabs, 'counts_by_labels': counts_by_labels, 'vocab_by_labels': vocabs_by_labels}


def generate_stats_files():
    rets = probe_labels()
    with open('../data/analytic/counts-label.csv', 'w', encoding='utf-8-sig') as f:
        f.write('tag,label,frequency\n')
        for tag in rets['counts'].keys():
            for key, val in rets['counts'][tag].items():
                f.write('{},{},{}\n'.format(tag, key, val))

    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    rets = probe_vocabs()
    for tag in rets['counts'].keys():
        sorted_counts = sorted(rets['counts'][tag].items(), key=lambda x: x[1], reverse=True)
        with open('../data/analytic/counts-vocab-{}.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            f.write('word,frequency\n')
            for key, val in sorted_counts:
                f.write('{},{}\n'.format(key, val))

        for label in labels:
            sorted_counts = sorted(rets['counts_by_labels'][tag][label].items(), key=lambda x: x[1], reverse=True)
            with open('../data/analytic/counts-vocab-{}-{}.csv'.format(tag, label), 'w', encoding='utf-8-sig') as f:
                f.write('word,frequency\n')
                for key, val in sorted_counts:
                    f.write('{},{}\n'.format(key, val))

    rets = probe_sentence_length()
    for tag in rets['counts'].keys():
        max_len = max(rets['counts'][tag].keys())
        min_len = min(rets['counts'][tag].keys())
        print('[{}]max: {}, min: {}'.format(tag, max_len, min_len))
        with open('../data/analytic/counts-sentence_length-{}.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            f.write('sentence length,all,{}\n'.format(','.join(labels)))
            for i in range(min_len, max_len+1, 1):
                line = []
                if i in rets['counts'][tag].keys():
                    line.append('{},{}'.format(i, len(rets['counts'][tag][i])))
                else:
                    line.append('{},{}'.format(i, 0))
                for label in labels:
                    if i in rets['counts_by_labels'][tag][label].keys():
                        line.append('{}'.format(rets['counts_by_labels'][tag][label][i]))
                    else:
                        line.append('{}'.format(0))
                f.write('{}\n'.format(','.join(line)))


def visualize_stats():
    tags = ['train', 'valid']
    labels = ['INFORMATIVE', 'UNINFORMATIVE']

    df = pd.DataFrame.from_csv('../data/analytic/counts-label.csv', encoding='utf-8-sig').reset_index()
    sns.barplot(x='label', y='frequency', data=df, hue='tag')
    plt.show()


if __name__ == '__main__':
    generate_stats_files()
    visualize_stats()