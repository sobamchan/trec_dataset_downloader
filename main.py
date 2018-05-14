import requests
import nltk


LABEL2ID = {
        'ABBR': 0,
        'DESC': 1,
        'ENTY': 2,
        'HUM': 3,
        'LOC': 4,
        'NUM': 5,
        }


def main():
    train_urls = [
            # 'http://cogcomp.org/Data/QA/QC/train_1000.label',
            # 'http://cogcomp.org/Data/QA/QC/train_2000.label',
            # 'http://cogcomp.org/Data/QA/QC/train_3000.label',
            # 'http://cogcomp.org/Data/QA/QC/train_4000.label',
            'http://cogcomp.org/Data/QA/QC/train_5500.label',
            ]
    test_url = 'http://cogcomp.org/Data/QA/QC/TREC_10.label'

    train_sents = []
    train_labels = []
    for train_url in train_urls:
        res = requests.get(train_url)
        for line in res.text.split('\n'):
            if len(line) <= 1:
                continue
            label_block = line[:line.find(' ')]
            label1, label2 = label_block.split(':')
            sent = line[line.find(' ')+1:]
            train_sents.append(' '.join(nltk.word_tokenize(sent.lower())))
            train_labels.append(LABEL2ID[label1])

    test_sents = []
    test_labels = []
    res = requests.get(test_url)
    for line in res.text.split('\n'):
        if len(line) <= 1:
            continue
        label_block = line[:line.find(' ')]
        label1, label2 = label_block.split(':')
        sent = line[line.find(' ')+1:]
        test_sents.append(' '.join(nltk.word_tokenize(sent)))
        test_labels.append(LABEL2ID[label1])
    return train_sents, train_labels, test_sents, test_labels
