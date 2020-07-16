import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as nltk_stopwords
from wordcloud import WordCloud

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

    n_grams = [1, 2, 3]
    raw_texts = datasets
    multi_stats = {i: {
        'vocabs': {tag: set() for tag in tags},
        'counts': {tag: {} for tag in tags},
        'vocabs_by_labels': {tag: {'INFORMATIVE': set(), 'UNINFORMATIVE': set()} for tag in tags},
        'counts_by_labels': {tag: {'INFORMATIVE': {}, 'UNINFORMATIVE': {}} for tag in tags},
        'ann_texts': {tag: [] for tag in tags},
        'del_texts': {tag: [] for tag in tags}
    } for i in n_grams}

    del_items = set(
        ['<hashtag>', '</hashtag>', '<allcaps>', '</allcaps>', '<user>', 'covid19', 'coronavirus', 'covid',
         '<number>', 'httpurl', 19, '19'])
    del_items |= set(["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", "<", ">", "(", ")", "/"])
    del_items |= set(nltk_stopwords.words('english'))
    for n_gram in n_grams:
        for tag in tags:
            for text, label in datasets[tag]:
                words = indexer.text_processor.pre_process_doc(text)
                label = get_label_text(label)
                multi_stats[n_gram]['ann_texts'][tag].extend([['_'.join(words[i: i+n_gram]) for i in range(0, len(words) - n_gram + 1)]])
                del_words = [word for word in words if word not in del_items]
                multi_stats[n_gram]['del_texts'][tag].extend([['_'.join(del_words[i: i+n_gram]) for i in range(0, len(del_words) - n_gram + 1)]])
                if n_gram != 1:
                    words = del_words
                for word in ['_'.join(words[i: i+n_gram]) for i in range(0, len(words) - n_gram + 1)]:
                    multi_stats[n_gram]['vocabs'][tag].add(word)
                    multi_stats[n_gram]['vocabs_by_labels'][tag][label].add(word)
                    if word in multi_stats[n_gram]['counts'][tag].keys():
                        multi_stats[n_gram]['counts'][tag][word] += 1
                    else:
                        multi_stats[n_gram]['counts'][tag][word] = 1
                    if word in multi_stats[n_gram]['counts_by_labels'][tag][label].keys():
                        multi_stats[n_gram]['counts_by_labels'][tag][label][word] += 1
                    else:
                        multi_stats[n_gram]['counts_by_labels'][tag][label][word] = 1
    return {'multi_stats': multi_stats, 'raw_texts': raw_texts}


def generate_stats_files():
    rets = probe_labels()
    with open('../data/analytic/counts-label.csv', 'w', encoding='utf-8-sig') as f:
        f.write('tag, label, frequency\n')
        for tag in rets['counts'].keys():
            for key, val in rets['counts'][tag].items():
                f.write('{}, {}, {}\n'.format(tag, key, val))

    all_count_table = {}
    stop_words = set(nltk_stopwords.words('english'))
    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    rets = probe_vocabs()
    visualize_wordcloud(rets['multi_stats'][1])

    rets = rets['multi_stats']
    for n_gram in rets.keys():
        vocab = rets[n_gram]['vocabs']['train'] | rets[n_gram]['vocabs']['valid']
        with open('../data/analytic/counts-vocab-{}_gram.csv'.format(n_gram), 'w', encoding='utf-8-sig') as f:
            f.write('word, train-INFORMATIVE, train-UNINFORMATIVE, valid-INFORMATIVE, valid-UNINFORMATIVE\n')
            for word in vocab:
                items = [word]
                for tag in rets[n_gram]['counts'].keys():
                    for label in labels:
                        if word in rets[n_gram]['counts_by_labels'][tag][label]:
                            items.extend([rets[n_gram]['counts_by_labels'][tag][label][word]])
                        else:
                            items.extend([0])
                f.write(', '.join(map(str, items)))
                f.write('\n')

    rets = rets[1]
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


def get_wordcloud(texts, path):
    wordcloud = WordCloud(background_color='white',
                          stopwords=None,
                          max_words=2000,
                          # max_font_size=100,
                          random_state=42,
                          width=1600, height=800
                          )
    del_items = set(['<hashtag>', '</hashtag>', '<allcaps>', '</allcaps>', '<user>', 'covid19', 'coronavirus', 'covid', '<number>', 'httpurl', 19, '19'])
    del_items |= set(nltk_stopwords.words('english'))
    for item in del_items:
        texts.pop(item, None)
    wordcloud.generate_from_frequencies(texts)
    fig = plt.figure(dpi=500)
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(path + '.pdf', format='pdf')
    plt.close('all')



def visualize_wordcloud(texts):
    tags = ['train', 'valid']
    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    for tag in tags:
        for label in labels:
            get_wordcloud(texts=texts['counts_by_labels'][tag][label], path='../data/analytic/wordcloud-{}-{}'.format(tag, label))


def visualize_stats():
    path = '../data/analytic/counts-label'
    df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
    plt.figure()
    sns.barplot(x='label', y='frequency', data=df, hue='tag')
    plt.savefig(path + '.pdf', format='pdf')
    plt.close('all')

    tags = ['train', 'valid']
    labels = ['INFORMATIVE', 'UNINFORMATIVE']
    for tag in tags:
        for tag_stopword in ['', '-without_stopwords']:
            path = '../data/analytic/counts-vocab-{}{}'.format(tag, tag_stopword)
            df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
            plt.figure()
            plt.xticks(rotation=90, fontsize=3)
            sns.barplot(x='word', y='frequency', data=df[:200])
            plt.savefig(path + '.pdf', format='pdf')
            plt.close('all')

    for tag in tags:
        for label in labels:
            for tag_stopword in ['', '-without_stopwords']:
                path = '../data/analytic/counts-vocab-{}-{}{}'.format(tag, label, tag_stopword)
                df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
                plt.figure()
                plt.xticks(rotation=90, fontsize=3)
                sns.barplot(x='word', y='frequency', data=df[:200])
                plt.savefig(path + '.pdf', format='pdf')
                plt.close('all')

    for tag in tags:
        path = '../data/analytic/counts-sentence_length-{}'.format(tag)
        df = pd.DataFrame.from_csv(path + '.csv', encoding='utf-8-sig', sep=', ').reset_index()
        plt.figure()
        df.plot.bar(x='sentence length', y=['INFORMATIVE', 'UNINFORMATIVE'])
        plt.savefig(path + '.pdf', format='pdf')
        plt.close('all')


if __name__ == '__main__':
    generate_stats_files()
    visualize_stats()