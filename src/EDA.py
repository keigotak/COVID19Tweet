import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from Indexer import Indexer
from HelperFunctions import get_datasets, get_raw_datasets, get_label_text


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
    datasets, tags = get_datasets()
    counts = {tag: {} for tag in tags}
    counts_by_labels = {tag: {'INFORMATIVE': {}, 'UNINFORMATIVE': {}} for tag in tags}
    indexer = Indexer()
    for tag in tags:
        for text, label in datasets[tag]:
            words = indexer.text_processor.pre_process_doc(text)
            label = get_label_text(label)
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
    datasets, tags = get_datasets()
    indexer = Indexer(with_preprocess=False)
    vocabs = {tag: set() for tag in tags}
    counts = {tag: {} for tag in tags}
    vocabs_by_labels = {tag: {'INFORMATIVE': set(), 'UNINFORMATIVE': set()} for tag in tags}
    counts_by_labels = {tag: {'INFORMATIVE': {}, 'UNINFORMATIVE': {}} for tag in tags}

    for tag in tags:
        for text, label in datasets[tag]:
            words = indexer.text_processor.pre_process_doc(text)
            label = get_label_text(label)
            for word in words:
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
        f.write('tag, label, frequency\n')
        for tag in rets['counts'].keys():
            for key, val in rets['counts'][tag].items():
                f.write('{}, {}, {}\n'.format(tag, key, val))

    all_count_table = {}
    stop_words = set(stopwords.words('english'))
    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    rets = probe_vocabs()
    for tag in rets['counts'].keys():
        sorted_counts = sorted(rets['counts'][tag].items(), key=lambda x: (x[1], x[0]), reverse=True)
        with open('../data/analytic/counts-vocab-{}.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            f.write('word, frequency\n')
            for key, val in sorted_counts:
                f.write('{}, {}\n'.format(key, val))
                table_key = '{}-ALL'.format(tag)
                if table_key not in all_count_table.keys():
                    all_count_table[table_key] = [key]
                else:
                    all_count_table[table_key].extend([key])

        with open('../data/analytic/counts-vocab-{}-without_stopwords.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            f.write('word, frequency\n')
            for key, val in sorted_counts:
                if key not in stop_words:
                    f.write('{}, {}\n'.format(key, val))
                    table_key = '{}-ALL-without_stopwords'.format(tag)
                    if table_key.format(tag) not in all_count_table.keys():
                        all_count_table[table_key] = [key]
                    else:
                        all_count_table[table_key].extend([key])


        for label in labels:
            sorted_counts = sorted(rets['counts_by_labels'][tag][label].items(), key=lambda x: (x[1], x[0]), reverse=True)
            with open('../data/analytic/counts-vocab-{}-{}.csv'.format(tag, label), 'w', encoding='utf-8-sig') as f:
                f.write('word, frequency\n')
                for key, val in sorted_counts:
                    f.write('{}, {}\n'.format(key, val))
                    table_key = '{}-{}'.format(tag, label)
                    if table_key not in all_count_table.keys():
                        all_count_table[table_key] = [key]
                    else:
                        all_count_table[table_key].extend([key])


            with open('../data/analytic/counts-vocab-{}-{}-without_stopwords.csv'.format(tag, label), 'w', encoding='utf-8-sig') as f:
                f.write('word, frequency\n')
                for key, val in sorted_counts:
                    if key not in stop_words:
                        f.write('{}, {}\n'.format(key, val))
                        table_key = '{}-{}-without_stopwords'.format(tag, label)
                        if table_key not in all_count_table.keys():
                            all_count_table[table_key] = [key]
                        else:
                            all_count_table[table_key].extend([key])

    for tag_stopword in ['', '-without_stopwords']:
        with open('../data/analytic/counts-vocab-ranking{}.csv'.format(tag_stopword), 'w', encoding='utf-8-sig') as f:
            headers = []
            max_length = 0
            for label in ['ALL'] + labels:
                for tag in rets['counts'].keys():
                    table_key = '{}-{}'.format(tag, label)
                    headers.extend([table_key])
                    max_length = max(len(all_count_table[table_key]), max_length)
            f.write(', '.join(headers) + '\n')
            for i in range(max_length):
                contents = []
                for label in ['ALL'] + labels:
                    for tag in rets['counts'].keys():
                        table_key = '{}-{}'.format(tag, label, tag_stopword)
                        if i < len(all_count_table[table_key]):
                            contents.extend([all_count_table[table_key][i]])
                        else:
                            contents.extend([''])
                f.write(', '.join(contents) + '\n')

    rets = probe_sentence_length()
    for tag in rets['counts'].keys():
        max_len = max(rets['counts'][tag].keys())
        min_len = min(rets['counts'][tag].keys())
        print('[{}]max: {}, min: {}'.format(tag, max_len, min_len))
        with open('../data/analytic/counts-sentence_length-{}.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            f.write('sentence length, all, {}\n'.format(', '.join(labels)))
            for i in range(min_len, max_len+1, 1):
                line = []
                if i in rets['counts'][tag].keys():
                    line.append('{}, {}'.format(i, len(rets['counts'][tag][i])))
                else:
                    line.append('{}, {}'.format(i, 0))
                for label in labels:
                    if i in rets['counts_by_labels'][tag][label].keys():
                        line.append('{}'.format(rets['counts_by_labels'][tag][label][i]))
                    else:
                        line.append('{}'.format(0))
                f.write('{}\n'.format(', '.join(line)))


def visualize_stats():
    path = '../data/analytic/counts-label'
    df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
    plt.figure()
    sns.barplot(x='label', y='frequency', data=df, hue='tag')
    plt.savefig(path + '.png')
    plt.close('all')

    tags = ['train', 'valid']
    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    for tag in tags:
        for tag_stopword in ['', '-without_stopwords']:
            path = '../data/analytic/counts-vocab-{}{}'.format(tag, tag_stopword)
            df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
            plt.figure()
            sns.barplot(x='word', y='frequency', fontsize=5, data=df[:200])
            plt.savefig(path + '.png')
            plt.close('all')

    for tag in tags:
        for label in labels:
            for tag_stopword in ['', '-without_stopwords']:
                path = '../data/analytic/counts-vocab-{}-{}{}'.format(tag, label, tag_stopword)
                df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
                plt.figure()
                sns.barplot(x='word', y='frequency', fontsize=5, data=df[:200])
                plt.savefig(path + '.png')
                plt.close('all')

    for tag in tags:
        path = '../data/analytic/counts-sentence_length-{}'.format(tag)
        df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
        plt.figure()
        df.plot.bar(x='sentence length', y=['INFORMATIVE', 'UNINFORMATIVE'])
        plt.savefig(path + '.png')
        plt.close('all')


if __name__ == '__main__':
    generate_stats_files()
    # visualize_stats()