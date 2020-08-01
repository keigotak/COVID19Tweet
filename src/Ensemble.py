from pathlib import Path
from statistics import mean


def get_path():
    base_path = Path('../data/results/details')
    files = [
        'details-20200731220126-epoch1.csv',
        'details-20200731220448-epoch1.csv',
        'details-20200731220708-epoch1.csv'
    ]
    return base_path, files


def get_average_score(line):
    sum_scores = []
    for item in line[2:]:
        if item == '':
            pass
        else:
            sum_scores.append(float(item))
    sum_score = mean(sum_scores) if len(sum_scores) != 0.0 else 0.0
    return str(sum_score)


def get_ensemble_table():
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

    table = []
    for sentence in sentences:
        line = [sentence, labels[files[0]][sentence]]
        for file in files:
            if sentence not in preds[file].keys():
                line.append('')
            else:
                line.append(preds[file][sentence])
        line = [get_average_score(line)] + line
        table.append(line)

    return table


def save_table(table, tag='tmp'):
    with Path('../data/results/details/{}.csv'.format(tag)).open('w', encoding='utf-8-sig') as f:
        for line in table:
            f.write('\t'.join(line))
            f.write('\n')


def main():
    table = get_ensemble_table()
    save_table(table, tag='ens1')


if __name__ == '__main__':
    main()

'''
データセットに重複がある．
# 1236997926888656897	In eastern China's Anhui province, all #COVID19 patients have been discharged after being cured. The recovery rate in the province has been 99.4% : HTTPURL HTTPURL	UNINFORMATIVE
# 1236997926888656897	In eastern China's Anhui province, all #COVID19 patients have been discharged after being cured. The recovery rate in the province has been 99.4% : HTTPURL HTTPURL	UNINFORMATIVE


'''



