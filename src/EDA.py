from HelperFunctions import get_raw_datasets


def probe_sentence_length():
    datasets, tags, path = get_raw_datasets()

    counts = {tag: {} for tag in tags}

    for tag in tags:
        for id, text, label in datasets[tag]:
            words = text.split(' ')
            if len(words) in counts[tag].keys():
                counts[tag][len(words)].append(words)
            else:
                counts[tag][len(words)] = [words]
    return counts

def probe_stats():
    datasets, tags, path = get_raw_datasets()

    vocabs = {tag: set() for tag in tags}
    counts = {tag: {} for tag in tags}

    for tag in tags:
        for id, text, label in datasets[tag]:
            words = text.split(' ')
            for word in words:
                word = word.lower()
                vocabs[tag].add(word)
                if word in counts[tag].keys():
                    counts[tag][word] += 1
                else:
                    counts[tag][word] = 1
        print(counts[tag])
    return counts, vocabs


def generate_stats_files():
    counts, vocabs = probe_stats()
    for tag in counts.keys():
        sorted_counts = sorted(counts[tag].items(), key=lambda x: x[1], reverse=True)
        with open('../data/analytic/counts-{}.csv'.format(tag), 'w', encoding='utf-8-sig') as f:
            for key, val in sorted_counts:
                f.write('{},{}\n'.format(key, val))

    counts = probe_sentence_length()
    for tag in counts.keys():
        max_len = max(counts[tag].keys())
        min_len = min(counts[tag].keys())
        print('[{}]max: {}, min: {}'.format(tag, max_len, min_len))
        for i in range(min_len, max_len+1, 1):
            if i in counts[tag].keys():
                print('{}: {}'.format(i, len(counts[tag][i])))
            else:
                print('{}: {}'.format(i, 0))


def main():
    generate_stats_files()



if __name__ == '__main__':
    main()